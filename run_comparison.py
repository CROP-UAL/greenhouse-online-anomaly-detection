"""Command-line entry points for the controlled physical-data experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.contract import ROOT, validate_splits
from shared.data_preparation.verify import audit_raw_lineage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("hydrate", "verify-data", "export-lightgbm", "train-lightgbm-night", "export-lightgbm-night",
                 "train-lstm", "export-lstm", "evaluate", "benchmark-lightgbm", "benchmark-lightgbm-night", "benchmark-lstm"):
        command = sub.add_parser(name)
        if name == "hydrate":
            command.add_argument("--lightgbm-source", type=Path, required=True)
        if name in ("export-lstm", "benchmark-lstm", "export-lightgbm-night", "benchmark-lightgbm-night"):
            command.add_argument("--run", type=Path, required=True)
        if name.startswith("benchmark"):
            command.add_argument("--rows", type=int, default=2000)
            command.add_argument("--repeats", type=int, default=5)
        if name == "evaluate":
            command.add_argument("--lightgbm-reference", action="store_true",
                                 help="Use the frozen 66-feature v8 run; default uses the night/context variant")
    args = parser.parse_args()
    if args.command == "hydrate":
        from shared.data_preparation.hydrate import hydrate
        print(json.dumps(hydrate(args.lightgbm_source), ensure_ascii=False, indent=2))
    elif args.command == "verify-data":
        print(json.dumps({"contract": validate_splits(), "raw_lineage": audit_raw_lineage()}, ensure_ascii=False, indent=2))
    elif args.command == "export-lightgbm":
        from approaches.lightgbm.adapter import export_test
        print(*export_test(), sep="\n")
    elif args.command == "train-lightgbm-night":
        from approaches.lightgbm.night_context import train
        print(train())
    elif args.command == "export-lightgbm-night":
        from approaches.lightgbm.night_context import export_test
        print(*export_test(args.run), sep="\n")
    elif args.command == "train-lstm":
        from approaches.lstm.adapter import train
        print(train())
    elif args.command == "export-lstm":
        from approaches.lstm.adapter import export_test
        print(*export_test(args.run), sep="\n")
    elif args.command == "evaluate":
        from shared.evaluation.evaluate import evaluate_pair
        traces = ROOT / "results" / "traces"
        prefix = "lightgbm" if args.lightgbm_reference else "lightgbm_night"
        report = evaluate_pair(traces / f"{prefix}_test.parquet", traces / f"{prefix}_model2_isolated_test.parquet",
                               traces / "lstm_test.parquet", traces / "lstm_model2_isolated_test.parquet")
        print(json.dumps({k: {"model_version": v["model_version"], "false_alarms_model1": v["false_alarms_model1"],
                              "missed_anomalies_model1": v["missed_anomalies_model1"]}
                          for k, v in report.items() if k in ("lightgbm", "lstm")}, ensure_ascii=False, indent=2))
    elif args.command == "benchmark-lightgbm":
        from approaches.lightgbm.benchmark import run_benchmark
        print(run_benchmark(rows=args.rows, repeats=args.repeats))
    elif args.command == "benchmark-lightgbm-night":
        from approaches.lightgbm.night_benchmark import run_benchmark
        print(run_benchmark(args.run, rows=args.rows, repeats=args.repeats))
    elif args.command == "benchmark-lstm":
        from approaches.lstm.benchmark import run_benchmark
        print(run_benchmark(args.run, rows=args.rows, repeats=args.repeats))


if __name__ == "__main__":
    main()
