#!/usr/bin/env python3
"""Run independent dataset processing and reporting for all sources."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[process_all_datasets] {message}")


def remove_named_artifacts(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and ("sid" in path.name.lower() or "semantic" in path.name.lower()):
            path.unlink()


def run_step(args: list[str]) -> None:
    log("running: " + " ".join(args))
    subprocess.run(args, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/processed",
        help="Generated dataset directory (recreated on each run).",
    )
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/data_report")
    parser.add_argument("--split-ratio", default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        log(f"remove existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    if args.report_dir.exists():
        log(f"remove existing report directory: {args.report_dir}")
        shutil.rmtree(args.report_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    remove_named_artifacts(args.output_dir)
    remove_named_artifacts(args.report_dir.parent)

    common = [
        "--raw-dir",
        str(args.raw_dir),
        "--output-dir",
        str(args.output_dir),
        "--report-dir",
        str(args.report_dir),
        "--split-ratio",
        args.split_ratio,
        "--seed",
        str(args.seed),
    ]
    if args.max_rows is not None:
        common.extend(["--max-rows", str(args.max_rows)])

    run_step([sys.executable, str(ROOT / "scripts/data/process_llm4poi.py"), *common])
    run_step([sys.executable, str(ROOT / "scripts/data/process_yelp.py"), *common])
    run_step([sys.executable, str(ROOT / "scripts/data/process_mobilitybench.py"), *common])
    run_step([sys.executable, str(ROOT / "scripts/data/analyze_all_datasets.py"), *common])
    remove_named_artifacts(args.output_dir)
    remove_named_artifacts(args.report_dir.parent)

    report = args.report_dir / "dataset_report.md"
    summary = {
        "output_dir": str(args.output_dir),
        "report": str(report),
    }
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
