"""LightGBM trained and evaluated on the same physical splits as LSTM."""
from __future__ import annotations
from datetime import datetime, timezone
import json, platform
from pathlib import Path
from time import perf_counter_ns

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from config import RANGOS_FISICOS
from src.artifacts import file_sha256
from src.binary_postprocess import (
    RULE_METHODS, RULE_NONE, RuleState, FrozenCalibration, binary_metrics,
    calibrate_binary, configured_rules, debounce_one, debounce_probabilities,
    rule_codes, rule_one,
)
from src.multiclass_features import EXTRA_FEATURES, HIGH, LOW, SharedFeatureExtractor
from seis_anomalias.data import FEATURES, SENSORS, decorate_masks, stream_matrix
from seis_anomalias.settings import BASE_FEATURES, CLASSES, MASK_COLUMNS

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent.parent
INPUT = REPOSITORY / "Dataset" / "MismoTest"
PARQUETS = {
    "train": INPUT / "train_injected_physical.parquet",
    "val": INPUT / "validation_injected_physical.parquet",
    "test": INPUT / "test_injected_physical.parquet",
}
RUNS = PACKAGE / "data" / "runs"
SCHEMA, HISTORY_ROWS = "lstm-common-lightgbm-v2", 360
COS_HOUR_INDEX = FEATURES.index("cos_hour")
DAY_NS = 86_400_000_000_000
PLOT_START = pd.Timestamp("2020-12-21 09:09:46")
PLOT_END = PLOT_START + pd.Timedelta(hours=4)
CATEGORIES = {
    "Sensor Atascado": ("Stuck Sensor", "#E69F00", "D", 7, 28),
    "Ruido": ("Noise", "#56B4E9", "^", 6, 28),
    "Desviacion de Correlacion": ("Correlation Deviation", "#009E73", "s", 5, 24),
    "Datos Faltantes": ("Missing Data", "#CC79A7", "v", 4, 26),
    "Valores Fuera de Rango": ("Out of Range", "#D55E00", "P", 8, 30),
    "Contextual": ("Contextual", "#0072B2", "o", 3, 22),
}

def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

def _json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def audit_inputs():
    audit, previous = {}, None
    required = {"FECHA", "tipo_anomalia", *SENSORS}
    for split, path in PARQUETS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{split}: missing {sorted(missing)}")
        dates = pd.to_datetime(df.FECHA)
        if dates.dt.tz is not None or dates.isna().any() or not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError(f"{split}: invalid FECHA")
        if previous is not None and dates.iloc[0] <= previous:
            raise ValueError(f"{split}: temporal overlap")
        previous = dates.iloc[-1]
        unknown = set(df.tipo_anomalia.dropna().astype(str)) - set(CLASSES)
        if unknown:
            raise ValueError(f"{split}: unknown labels {sorted(unknown)}")
        for flag in ("sequence_unsafe", "has_unresolved_nan"):
            if flag in df and df[flag].fillna(False).astype(bool).any():
                raise ValueError(f"{split}: {flag} has true values")
        audit[split] = {
            "path": str(path), "sha256": file_sha256(path), "rows": len(df),
            "columns": len(df.columns), "first_FECHA": str(dates.iloc[0]),
            "last_FECHA": str(dates.iloc[-1]), "duplicates": int(dates.duplicated().sum()),
            "labels": {str(k): int(v) for k, v in
                       df.tipo_anomalia.fillna("Normal").value_counts().items()},
        }
    return audit

def _frames(include_test=False):
    audit_inputs()
    names = ("train", "val", "test") if include_test else ("train", "val")
    raw = {k: pd.read_parquet(PARQUETS[k]) for k in names}
    nominal = pd.date_range("2020-10-10", periods=sum(map(len, raw.values())), freq="30s")
    out, offset = {}, 0
    for split in names:
        df = raw[split].loc[:, ["FECHA", "tipo_anomalia", *SENSORS]].copy()
        df["FECHA"] = pd.to_datetime(df.FECHA)
        df["Fecha"] = nominal[offset:offset+len(df)]
        offset += len(df)
        df["tipo_anomalia"] = df.tipo_anomalia.fillna("Normal").astype(str)
        df["target_anomalia"] = df.tipo_anomalia.ne("Normal").astype(np.uint8)
        out[split] = decorate_masks(df)
    return out

def _data(include_test=False):
    frames = _frames(include_test=include_test)
    fills = np.zeros(len(BASE_FEATURES), np.float32)
    model_frames = {
        "train": frames["train"].iloc[HISTORY_ROWS:].copy(),
        "val": frames["val"],
    }
    matrices = {
        "train": stream_matrix(model_frames["train"], frames["train"].iloc[:HISTORY_ROWS], fills),
        "val": stream_matrix(frames["val"], frames["train"].iloc[-HISTORY_ROWS:], fills),

    }
    if include_test:
        model_frames["test"] = frames["test"]
        matrices["test"] = stream_matrix(frames["test"], frames["val"].iloc[-HISTORY_ROWS:], fills)
    for split, matrix in matrices.items():
        physical_ns = model_frames[split].FECHA.to_numpy("datetime64[ns]").astype(np.int64)
        matrix[:, COS_HOUR_INDEX] = np.cos(2*np.pi*(physical_ns % DAY_NS)/DAY_NS).astype(np.float32)
    for split in matrices:
        if matrices[split].shape != (len(model_frames[split]), len(FEATURES)):
            raise ValueError(f"{split}: feature shape mismatch")
        if not np.isfinite(matrices[split]).all():
            raise ValueError(f"{split}: nonfinite features")
    return frames, model_frames, matrices

def _rules(frame, history):
    raw = frame.loc[:, SENSORS].to_numpy(np.float32)
    old = history.loc[:, SENSORS].to_numpy(np.float32)
    times = frame.Fecha.to_numpy("datetime64[ns]").astype(np.int64)
    old_times = history.Fecha.to_numpy("datetime64[ns]").astype(np.int64)
    policy = configured_rules()
    _, state = rule_codes(old, np.isnan(old).astype(np.uint8), old_times, policy, LOW, HIGH)
    codes, _ = rule_codes(raw, np.isnan(raw).astype(np.uint8), times, policy, LOW, HIGH, state=state)
    return codes

def _decide(model, calibration, matrix, frame, history):
    prob = np.asarray(model.predict(matrix, num_threads=1), np.float32)
    ml = debounce_probabilities(prob, calibration.thresh_std, calibration.k_long,
                                calibration.thresh_high, calibration.k_short)
    rules = _rules(frame, history)
    return prob, np.maximum(ml, (rules != RULE_NONE).astype(np.uint8)), rules

def train_binary():
    audit = audit_inputs()
    frames, table, matrices = _data()
    train, val = table["train"], table["val"]
    yt = train.target_anomalia.to_numpy(np.uint8)
    yv = val.target_anomalia.to_numpy(np.uint8)
    counts = np.bincount(yt, minlength=2)
    params = {"objective":"binary","metric":"auc","learning_rate":0.04,
        "num_leaves":31,"max_depth":6,"min_data_in_leaf":25,"feature_fraction":0.7,
        "verbosity":-1,"num_threads":1,"seed":42,"deterministic":True,
        "force_col_wise":True,"scale_pos_weight":float(counts[0]/max(counts[1],1))}
    dtrain = lgb.Dataset(matrices["train"], label=yt, feature_name=list(FEATURES))
    dval = lgb.Dataset(matrices["val"], label=yv, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=400, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
    prob = np.asarray(model.predict(matrices["val"], num_threads=1), np.float32)
    guards = (_rules(val, frames["train"].iloc[-HISTORY_ROWS:]) != RULE_NONE).astype(np.uint8)
    calibration, score = calibrate_binary(prob, yv, days=len(val)/2880,
        std_grid=np.asarray([.2,.3,.4,.5],np.float32),
        long_grid=np.asarray([2,3,4],np.int32),
        high_grid=np.asarray([.75,.85,.95],np.float32),
        short_grid=np.asarray([1,2],np.int32),
        max_false_alarms_per_day=10, forced_positive=guards)
    RUNS.mkdir(parents=True, exist_ok=True)
    run = RUNS / _stamp(); run.mkdir()
    model.save_model(str(run/"binary_model.txt"))
    calibration.save(run/"binary_calibration.json")
    _json(run/"binary_validation.json", {"metrics":score,"best_iteration":model.best_iteration,
          "train_counts":counts.tolist(),"val_counts":np.bincount(yv,minlength=2).tolist()})
    _json(run/"binary_manifest.json", {"schema_version":SCHEMA,"stage":"binary",
          "created_utc":datetime.now(timezone.utc).isoformat(),"input_audit":audit,
          "feature_clock":"30 s by row for causal continuity; cos_hour from physical FECHA","original_timestamp":"FECHA",
          "history_rows":HISTORY_ROWS,"features":list(FEATURES),"classes":list(CLASSES),
          "artifacts":{n:file_sha256(run/n) for n in
              ("binary_model.txt","binary_calibration.json","binary_validation.json")}})
    return run

def _load_binary(run):
    run=Path(run); manifest=_read(run/"binary_manifest.json")
    if manifest.get("schema_version") != SCHEMA or tuple(manifest["features"]) != FEATURES:
        raise ValueError("Incompatible binary manifest")
    current=audit_inputs()
    if {k:v["sha256"] for k,v in current.items()} != {k:v["sha256"] for k,v in manifest["input_audit"].items()}:
        raise ValueError("Input parquet hashes changed")
    for n,d in manifest["artifacts"].items():
        if file_sha256(run/n) != d: raise ValueError(f"Hash mismatch: {n}")
    model=lgb.Booster(model_file=str(run/"binary_model.txt"))
    if model.num_feature()!=len(FEATURES) or tuple(model.feature_name())!=FEATURES:
        raise ValueError("Binary feature order mismatch")
    return manifest,model,FrozenCalibration.load(run/"binary_calibration.json")

def train_multiclass(run):
    run=Path(run); binary_manifest,_,_=_load_binary(run)
    if (run/"manifest.json").exists(): raise FileExistsError(run/"manifest.json")
    _,table,matrices=_data()
    chosen={}; y={}
    for split in ("train","val"):
        chosen[split]=table[split].target_anomalia.to_numpy(np.uint8).astype(bool)
        y[split]=np.asarray([CLASSES.index(x) for x in table[split].loc[chosen[split],"tipo_anomalia"]],np.int32)
    counts=np.bincount(y["train"],minlength=len(CLASSES))
    if np.any(counts==0): raise ValueError("Training lacks a class")
    weights=len(y["train"])/(len(CLASSES)*counts)
    params={"objective":"multiclass","num_class":len(CLASSES),"metric":"multi_logloss",
      "learning_rate":.05,"num_leaves":31,"max_depth":6,"min_data_in_leaf":30,
      "feature_fraction":.9,"verbosity":-1,"seed":42,"deterministic":True,
      "force_col_wise":True,"num_threads":1}
    dt=lgb.Dataset(matrices["train"][chosen["train"]],label=y["train"],
                   weight=weights[y["train"]],feature_name=list(FEATURES))
    dv=lgb.Dataset(matrices["val"][chosen["val"]],label=y["val"],reference=dt)
    model=lgb.train(params,dt,num_boost_round=500,valid_sets=[dv],
                    callbacks=[lgb.early_stopping(50,verbose=False)])
    scores=np.asarray(model.predict(matrices["val"][chosen["val"]],num_threads=1))
    model.save_model(str(run/"multiclass_model.txt"))
    _json(run/"multiclass_validation.json",{"macro_f1":float(f1_score(y["val"],scores.argmax(1),
          labels=range(len(CLASSES)),average="macro",zero_division=0)),
          "best_iteration":model.best_iteration,"class_weights":dict(zip(CLASSES,weights.tolist())),
          "support":{s:dict(zip(CLASSES,np.bincount(y[s],minlength=len(CLASSES)).tolist())) for s in y}})
    _json(run/"manifest.json",{"schema_version":SCHEMA,"stage":"complete",
          "classes":list(CLASSES),"features":list(FEATURES),
          "binary_manifest_sha256":file_sha256(run/"binary_manifest.json"),
          "input_hashes":{k:v["sha256"] for k,v in binary_manifest["input_audit"].items()},
          "artifacts":{n:file_sha256(run/n) for n in ("multiclass_model.txt","multiclass_validation.json")}})
    return run

def load_run(run):
    run=Path(run); _,binary,calibration=_load_binary(run); manifest=_read(run/"manifest.json")
    if manifest.get("schema_version")!=SCHEMA or tuple(manifest["classes"])!=CLASSES or tuple(manifest["features"])!=FEATURES:
        raise ValueError("Incompatible complete manifest")
    if manifest["binary_manifest_sha256"]!=file_sha256(run/"binary_manifest.json"):
        raise ValueError("Binary manifest changed")
    for n,d in manifest["artifacts"].items():
        if file_sha256(run/n)!=d: raise ValueError(f"Hash mismatch: {n}")
    multi=lgb.Booster(model_file=str(run/"multiclass_model.txt"))
    return binary,calibration,multi

def evaluate(run):
    run=Path(run); binary,calibration,multi=load_run(run)
    frames,table,matrices=_data(include_test=True); test=table["test"]
    prob,detected,rules=_decide(binary,calibration,matrices["test"],test,frames["val"].iloc[-HISTORY_ROWS:])
    routed=np.flatnonzero(detected); pred=np.full(len(test),"Normal",object)
    scores=np.full((len(test),len(CLASSES)),np.nan,np.float32)
    if len(routed):
        scores[routed]=multi.predict(matrices["test"][routed],num_threads=1)
        pred[routed]=np.asarray(CLASSES,object)[scores[routed].argmax(1)]
    truth=test.tipo_anomalia.to_numpy(); y=test.target_anomalia.to_numpy(np.uint8)
    labels=("Normal",)+CLASSES
    methods=[RULE_METHODS[int(c)] if c else ("lightgbm_debouncing" if h else "normal") for c,h in zip(rules,detected)]
    trace=pd.DataFrame({"FECHA":test.FECHA.to_numpy(),"PRAD":test.PRAD.to_numpy(),
      "target_anomalia":y,"tipo_real":truth,"probabilidad_binaria":prob,
      "pred_binaria":detected,"metodo_deteccion_binaria":methods,"tipo_predicho":pred})
    if len(trace)!=len(test) or not trace.FECHA.is_unique: raise ValueError("Trace alignment failed")
    report={"rows":len(trace),"true_anomalies":int(y.sum()),"routed_to_multiclass":len(routed),
      "without_multiclass_prediction":len(trace)-len(routed),
      "binary":binary_metrics(y,detected,len(test)/2880),"labels":list(labels),
      "end_to_end_macro_f1":float(f1_score(truth,pred,labels=labels,average="macro",zero_division=0)),
      "end_to_end_confusion_matrix":confusion_matrix(truth,pred,labels=labels).tolist(),
      "end_to_end_report":classification_report(truth,pred,labels=labels,output_dict=True,zero_division=0),
      "predicted_type_counts":trace.tipo_predicho.value_counts().to_dict(),
      "first_FECHA":str(trace.FECHA.iloc[0]),"last_FECHA":str(trace.FECHA.iloc[-1])}
    result=run/"results"/_stamp(); result.mkdir(parents=True)
    trace.to_parquet(result/"test_trace.parquet",index=False)
    np.savez_compressed(result/"type_probabilities.npz",scores=scores,classes=np.asarray(CLASSES))
    _json(result/"metrics.json",report)
    _json(result/"result_manifest.json",{"schema_version":SCHEMA,
      "run_manifest_sha256":file_sha256(run/"manifest.json"),
      "artifacts":{n:file_sha256(result/n) for n in ("test_trace.parquet","type_probabilities.npz","metrics.json")}})
    plot_prad(result)
    return result

def plot_prad(result):
    result=Path(result); trace=pd.read_parquet(result/"test_trace.parquet")
    w=trace[(trace.FECHA>=PLOT_START)&(trace.FECHA<PLOT_END)].copy()
    counts=w.loc[w.pred_binaria.eq(1),"tipo_predicho"].value_counts()
    fig,ax=plt.subplots(figsize=(12,5.5))
    ax.plot(w.FECHA,w.PRAD,color="#444444",linewidth=.8,alpha=.9,zorder=2,label="PRAD signal")
    for cat,count in counts.items():
        label,color,marker,zorder,size=CATEGORIES[cat]; marked=w[w.tipo_predicho.eq(cat)]
        ax.scatter(marked.FECHA,marked.PRAD,s=size,c=color,marker=marker,alpha=.88,
          linewidths=.3,edgecolors="white",zorder=zorder,label=f"{label} (n={count:,})")
    ax.set_title("(b) PRAD sensor — LightGBM predicted anomaly categories during a 4-hour period at 21-12-2020",
      loc="left",fontweight="bold",pad=10)
    ax.set_ylabel("Global Radiation [W m⁻²]"); ax.set_xlabel("Date / Time (UTC+1)")
    ax.set_xlim(PLOT_START,PLOT_END); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator()); ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0,60,15)))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    handles,names=ax.get_legend_handles_labels()
    items=[(h,n) for h,n in zip(handles,names) if n!="PRAD signal"]+[(h,n) for h,n in zip(handles,names) if n=="PRAD signal"]
    legend=ax.legend(*zip(*items),title="Predicted anomaly type",loc="upper right",
      framealpha=.92,edgecolor="#BBBBBB",handletextpad=.4,labelspacing=.4,borderpad=.6,ncol=2)
    legend._legend_box.align="left"; ax.grid(axis="y",linestyle=":",linewidth=.5,alpha=.35)
    fig.tight_layout(); stem=result/"fig_anomaly_detection_4h_lightgbm_lstm_common_2020-12-21"
    for suffix in (".png",".pdf"): fig.savefig(stem.with_suffix(suffix),dpi=300,bbox_inches="tight")
    plt.close(fig); missing=w.PRAD.isna()
    summary={"window":f"[{PLOT_START}, {PLOT_END})","rows":len(w),
      "first_FECHA":str(w.FECHA.iloc[0]),"last_FECHA":str(w.FECHA.iloc[-1]),
      "predicted_type_counts":counts.to_dict(),"missing_PRAD":int(missing.sum()),
      "routed_markers_without_PRAD":int((missing&w.pred_binaria.eq(1)).sum())}
    if not (result/"figure_summary.json").exists(): _json(result/"figure_summary.json",summary)
    return summary

def benchmark(run,rows=None,repeats=5):
    run=Path(run); binary,calibration,multi=load_run(run); frames=_frames(include_test=True)
    test=frames["test"]; history=frames["val"].iloc[-HISTORY_ROWS:]
    n=len(test) if rows is None else min(int(rows),len(test))
    raw=np.ascontiguousarray(test.loc[:,SENSORS].to_numpy(np.float32)[:n])
    times=test.Fecha.to_numpy("datetime64[ns]").astype(np.int64)[:n]
    physical_times=test.FECHA.to_numpy("datetime64[ns]").astype(np.int64)[:n]
    masks=np.ascontiguousarray(test.loc[:,MASK_COLUMNS].to_numpy(np.uint8)[:n])
    old=np.ascontiguousarray(history.loc[:,SENSORS].to_numpy(np.float32))
    old_times=history.Fecha.to_numpy("datetime64[ns]").astype(np.int64)
    old_masks=np.ascontiguousarray(history.loc[:,MASK_COLUMNS].to_numpy(np.uint8))
    lo=np.asarray([RANGOS_FISICOS[s]["min"] for s in SENSORS],np.float32)
    hi=np.asarray([RANGOS_FISICOS[s]["max"] for s in SENSORS],np.float32)
    policy=configured_rules(); row_samples=[]; pass_ns=[]; routed_counts=[]; last=None
    for pass_index in range(repeats+1):
        extractor=SharedFeatureExtractor(BASE_FEATURES,np.zeros(len(BASE_FEATURES),np.float32),old,old_times,old_masks,times[0])
        rule_state=RuleState()
        for rv,tv in zip(old,old_times): rule_one(rv,np.isnan(rv).astype(np.uint8),int(tv),rule_state,policy,lo,hi)
        long_count=short_count=0; pred=np.empty(n,object); durations=np.empty(n,np.int64); routed=0
        begin=perf_counter_ns()
        for i in range(n):
            start=perf_counter_ns(); vector=np.empty(len(FEATURES),np.float32)
            vector[:len(BASE_FEATURES)+len(EXTRA_FEATURES)]=extractor.step(raw[i],times[i],masks[i])
            vector[-len(SENSORS):]=masks[i]
            vector[COS_HOUR_INDEX]=np.float32(np.cos(2*np.pi*(int(physical_times[i]) % DAY_NS)/DAY_NS))
            p=float(binary.predict(vector.reshape(1,-1),num_threads=1)[0])
            ml,long_count,short_count=debounce_one(p,calibration.thresh_std,calibration.k_long,
              calibration.thresh_high,calibration.k_short,long_count,short_count)
            rule=rule_one(raw[i],np.isnan(raw[i]).astype(np.uint8),int(times[i]),rule_state,policy,lo,hi)
            if ml or rule!=RULE_NONE:
                pred[i]=CLASSES[int(np.argmax(multi.predict(vector.reshape(1,-1),num_threads=1)[0]))]; routed+=1
            else: pred[i]="Normal"
            durations[i]=perf_counter_ns()-start
        if pass_index == 0:
            continue
        pass_ns.append(perf_counter_ns()-begin); row_samples.append(durations); routed_counts.append(routed)
        if last is not None and not np.array_equal(last,pred): raise ValueError("Benchmark is nondeterministic")
        last=pred.copy()
    traces=sorted((run/"results").glob("*/test_trace.parquet"))
    parity=None
    if traces:
        expected=pd.read_parquet(traces[-1],columns=["tipo_predicho"]).tipo_predicho.to_numpy()[:n]
        parity=bool(np.array_equal(last,expected))
        if not parity: raise ValueError("Sequential benchmark differs from exported test trace")
    ms=np.concatenate(row_samples)/1e6
    report={"schema_version":SCHEMA,"platform":platform.platform(),"machine":platform.machine(),
      "python":platform.python_version(),"lightgbm":lgb.__version__,"rows_per_pass":n,
      "repeats":repeats,"warmup_passes":1,"parity_with_test_trace":parity,"invocations_m2_per_pass":routed_counts,"mean_ms":float(ms.mean()),
      "p50_ms":float(np.percentile(ms,50)),"p95_ms":float(np.percentile(ms,95)),
      "p99_ms":float(np.percentile(ms,99)),
      "throughput_rows_per_second":float(repeats*n/(sum(pass_ns)/1e9)),
      "timing_scope":"causal features + binary + rules + conditional multiclass",
      "excluded":"model loading and parquet I/O","run_manifest_sha256":file_sha256(run/"manifest.json")}
    output=run/"benchmarks"/f"{_stamp()}.json"; output.parent.mkdir(parents=True,exist_ok=True); _json(output,report)
    return output

def latest_run():
    candidates=sorted(RUNS.glob("*/manifest.json"))
    if not candidates: raise FileNotFoundError("No complete lstm_common run")
    return candidates[-1].parent

