"""Fetch CSI 300 constituents and save them as a local universe CSV."""

from __future__ import annotations

from ashare_quant import write_hs300_constituents


def main() -> None:
    path = write_hs300_constituents()
    print(f"CSI 300 constituents written to: {path}")


if __name__ == "__main__":
    main()
