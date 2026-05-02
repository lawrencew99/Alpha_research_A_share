from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ashare_quant.config import load_config
from ashare_quant.pipeline import run_research


def main() -> None:
    config = load_config("configs/default.yaml")
    result = run_research(config, demo=True)
    print("Backtest metrics:")
    for key, value in result["backtest"]["metrics"].items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
