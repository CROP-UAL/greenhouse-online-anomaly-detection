import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from approaches.lstm.causal_data import M2_EXCLUDE, M2_REDUNDANT, STUCK_SENSORS, _causal_diagnostics, _stuck_run_lengths, windows
from approaches.lightgbm.night_context import ADDED_FEATURES, co2_gradient, night_flag
from approaches.lightgbm.v8_online_2meses.seis_anomalias.settings import BASE_FEATURES
from approaches.lstm.src.scaling import MODEL_FEATURES
from shared.contract import SPLITS, ROWS, load_split, validate_splits
from shared.data_preparation.verify import audit_raw_lineage
from shared.injection.replay import replay
from shared.trace import make_trace, validate_trace


def test_frozen_physical_contract_and_raw_lineage():
    contract = validate_splits()
    assert {name: contract["splits"][name]["rows"] for name in SPLITS} == ROWS
    assert audit_raw_lineage()["original_timestamp_sequence_matches"]


def test_replay_is_exact_and_repeatable():
    first, first_report = replay()
    second, second_report = replay()
    assert first_report["plan_sha256"] == second_report["plan_sha256"]
    for name in SPLITS:
        assert_frame_equal(first[name], second[name], check_dtype=False)
        assert_frame_equal(first[name], load_split(name).drop(columns=["tipo_canonico"]), check_dtype=False)


def test_lstm_windows_cross_splits_using_only_prior_rows():
    matrices = {"train": np.arange(25, dtype=np.float32)[:, None],
                "validation": np.arange(25, 30, dtype=np.float32)[:, None],
                "test": np.arange(30, 32, dtype=np.float32)[:, None]}
    x, rows = windows(matrices, "validation", targets=[0, 1], length=20)
    assert rows.tolist() == [0, 1]
    assert x[0, :, 0].tolist() == list(range(6, 26))
    y, test_rows = windows(matrices, "test", targets=[0], length=20)
    assert test_rows.tolist() == [0]
    assert y[0, :, 0].tolist() == list(range(11, 31))
    with np.testing.assert_raises(ValueError):
        windows(matrices, "train", targets=[0], length=20)
    physical = {name: np.arange(ROWS[name], dtype=np.float32)[:, None] for name in SPLITS}
    _, full_test_targets = windows(physical, "test")
    assert np.array_equal(full_test_targets, np.arange(ROWS["test"]))
    assert len(load_split("test")) == len(full_test_targets)


def test_causal_lstm_diagnostics_do_not_change_before_future_perturbation():
    from approaches.lstm.src.feature_engineering import engineer_features

    frame = load_split("train").head(45).copy()
    sensors = ["PCO2EXT", "PHEXT", "PRAD", "PRGINT", "PTEXT", "PVV", "UVENT_cen", "UVENT_lN", "XCO2I", "XHINV", "XTINV", "XTS"]
    frame.loc[:, sensors] = frame.loc[:, sensors].ffill().bfill()
    earlier = _causal_diagnostics(engineer_features(frame))
    frame.loc[44, "XTINV"] += 1000
    changed = _causal_diagnostics(engineer_features(frame))
    assert_frame_equal(earlier.iloc[:44], changed.iloc[:44])


def test_stuck_context_tracks_only_past_eligible_sensor_values():
    assert "PRAD" not in STUCK_SENSORS
    frame = pd.DataFrame({sensor: np.arange(7, dtype=float) for sensor in STUCK_SENSORS})
    frame["XCO2I"] = [1, 2, 2, 2, 3, 3, 4]
    features = _stuck_run_lengths(frame)
    assert features["XCO2I_stuck_run_length"].tolist() == [1, 1, 2, 3, 1, 2, 1]
    frame.loc[6, "XCO2I"] = 3
    changed = _stuck_run_lengths(frame)
    assert_frame_equal(features.iloc[:6], changed.iloc[:6])


def test_trace_routing_and_episode_ids():
    frame = load_split("test").head(12)
    binary = np.array([True] + [False] * 11)
    stage2 = np.array(["Noise"] + [None] * 11, dtype=object)
    trace = make_trace(frame, split="test", scores=np.full(12, 0.5), binary=binary, stage2=stage2,
                       model_version="synthetic-test", data_manifest_id="test-manifest")
    validate_trace(trace)
    assert trace.loc[0, "final_prediction"] == "Noise"
    assert trace.loc[1:, "final_prediction"].eq("Normal").all()


def test_both_feature_contracts_expose_night_and_inside_outside_context():
    times = pd.Series(pd.to_datetime(["2020-12-21 06:59:59", "2020-12-21 07:00:00",
                                      "2020-12-21 19:59:59", "2020-12-21 20:00:00"]))
    assert night_flag(times).tolist() == [1.0, 0.0, 0.0, 1.0]
    assert ADDED_FEATURES == ("is_night", "co2_gradient")
    assert co2_gradient(pd.DataFrame({"XCO2I": [450., 300.], "PCO2EXT": [400., 400.]})).tolist() == [50., -100.]
    assert {"delta_temp", "delta_hum", "diff_abs_XCO2I_PCO2EXT"}.issubset(BASE_FEATURES)
    assert {"delta_temp", "delta_hum", "co2_gradient", "is_night"}.issubset(MODEL_FEATURES)
    assert not {"delta_temp", "delta_hum", "co2_gradient", "is_night"} & (M2_EXCLUDE | M2_REDUNDANT)
