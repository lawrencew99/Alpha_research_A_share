"""Stock universe loaders for A-share research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_HS300_PATH = Path(__file__).resolve().parents[2] / "data" / "hs300_constituents.csv"


def normalize_ashare_ticker(code: str | int) -> str:
    """Normalize raw A-share code to the package ticker format, e.g. 600519.SH."""

    value = str(code).strip().upper()
    if value.endswith((".SH", ".SZ")):
        return value
    if value.startswith(("SH", "SZ")):
        return f"{value[2:]}.{value[:2]}"

    digits = value.zfill(6)
    suffix = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return f"{digits}.{suffix}"


def load_universe_csv(path: str | Path, code_column: str = "ticker") -> list[str]:
    """Load a ticker list from CSV."""

    frame = pd.read_csv(path, dtype={code_column: str})
    if code_column not in frame.columns:
        raise ValueError(f"column {code_column!r} was not found in {path}")
    return frame[code_column].map(normalize_ashare_ticker).drop_duplicates().tolist()


def load_hs300_constituents(path: str | Path = DEFAULT_HS300_PATH) -> list[str]:
    """Load saved CSI 300 constituents."""

    return load_universe_csv(path, code_column="ticker")


def fetch_hs300_constituents() -> pd.DataFrame:
    """Fetch current CSI 300 constituents with AKShare.

    Install the optional data dependency first:
    pip install "ashare-quant[data]"
    """

    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError('AKShare is required. Run from the project root: pip install -e ".[data]"') from exc

    try:
        raw = ak.index_stock_cons_weight_csindex(symbol="000300")
        result = _parse_csindex_weight_frame(raw)
    except Exception:
        raw = ak.index_stock_cons_sina(symbol="000300")
        result = _parse_sina_frame(raw)

    return result.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def write_hs300_constituents(path: str | Path = DEFAULT_HS300_PATH) -> Path:
    """Fetch and save current CSI 300 constituents to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fetch_hs300_constituents().to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _parse_csindex_weight_frame(raw: pd.DataFrame) -> pd.DataFrame:
    code_col = _first_existing_column(raw, ["成分券代码", "证券代码", "代码", "stock_code"])
    name_col = _first_existing_column(raw, ["成分券名称", "证券简称", "名称", "stock_name"])
    date_col = _first_existing_column(raw, ["日期", "date"])
    weight_col = _first_existing_column(raw, ["权重", "weight"])

    code = raw[code_col] if code_col else raw.iloc[:, 4]
    name = raw[name_col].astype(str) if name_col else raw.iloc[:, 5].astype(str)
    as_of_date = raw[date_col].astype(str) if date_col else raw.iloc[:, 0].astype(str)
    weight = pd.to_numeric(raw[weight_col] if weight_col else raw.iloc[:, -1], errors="coerce")

    return pd.DataFrame(
        {
            "ticker": code.map(normalize_ashare_ticker),
            "name": name,
            "weight": weight,
            "as_of_date": as_of_date,
        }
    )


def _parse_sina_frame(raw: pd.DataFrame) -> pd.DataFrame:
    code_col = _first_existing_column(raw, ["code", "symbol", "证券代码", "代码"])
    name_col = _first_existing_column(raw, ["name", "证券简称", "名称"])
    return pd.DataFrame(
        {
            "ticker": raw[code_col].map(normalize_ashare_ticker),
            "name": raw[name_col].astype(str) if name_col else "",
        }
    )
