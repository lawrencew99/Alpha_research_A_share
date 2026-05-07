"""Merge metrics.csv from several output directories into one summary table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="Pairs: output_dir run_label, e.g. outputs_top20 top_n_20 outputs_weekly rebalance_WFRI",
    )
    args = parser.parse_args()
    pairs = list(zip(args.paths[0::2], args.paths[1::2]))
    if len(pairs) * 2 != len(args.paths):
        print("Need an even number of arguments: dir label dir label ...", file=sys.stderr)
        sys.exit(1)

    rows: list[pd.Series] = []
    root = Path(__file__).resolve().parents[1]
    for out_dir, label in pairs:
        path = root / out_dir / "metrics.csv"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        frame = pd.read_csv(path, index_col=0)["value"]
        frame.name = label
        rows.append(frame)

    if not rows:
        sys.exit(2)

    summary = pd.DataFrame(rows)
    out = root / "outputs_sensitivity_summary.csv"
    summary.to_csv(out, encoding="utf-8-sig")
    print(summary.round(4).to_string())
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
