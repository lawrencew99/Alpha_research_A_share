"""Data loading, adjustment and A-share universe cleaning utilities."""

from __future__ import annotations

from pathlib import Path
from time import sleep

import pandas as pd


REQUIRED_PRICE_COLUMNS = {"date", "ticker", "close"}


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load local A-share OHLCV data and return a date/ticker indexed panel."""

    frame = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_PRICE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    frame = frame.sort_values(["date", "ticker"]).set_index(["date", "ticker"])
    return add_adjusted_prices(frame)


def load_akshare_ashare_history(
    tickers: list[str],
    start: str = "2021-01-01",
    end: str = "2024-12-31",
    adjust: str = "qfq",
    cache_dir: str | Path = "data/raw/akshare_daily",
    sleep_seconds: float = 0.5,
    max_retries: int = 3,
    max_workers: int = 1,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Load real A-share daily history from AKShare with local CSV caching."""

    panel, _ = load_akshare_ashare_history_with_skips(
        tickers=tickers,
        start=start,
        end=end,
        adjust=adjust,
        cache_dir=cache_dir,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    return panel


def load_akshare_ashare_history_with_skips(
    tickers: list[str],
    start: str = "2021-01-01",
    end: str = "2024-12-31",
    adjust: str = "qfq",
    cache_dir: str | Path = "data/raw/akshare_daily",
    sleep_seconds: float = 0.5,
    max_retries: int = 3,
    max_workers: int = 1,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load real A-share daily history and return skipped tickers separately."""

    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError('AKShare is required. Run from the project root: pip install -e ".[data]"') from exc

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    frames, failures, skipped = _load_akshare_histories(
        ak_module=ak,
        tickers=tickers,
        start=start,
        end=end,
        adjust=adjust,
        cache_dir=cache_path,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
        max_workers=max_workers,
        show_progress=show_progress,
    )

    if failures:
        failed = "\n".join(failures[:10])
        raise RuntimeError(f"AKShare failed for {len(failures)} tickers. First failures:\n{failed}")
    if not frames:
        raise RuntimeError("AKShare did not return usable daily data for any ticker.")

    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    panel = panel.set_index(["date", "ticker"])
    skipped_frame = pd.DataFrame(skipped, columns=["ticker", "reason"])
    return add_adjusted_prices(panel), skipped_frame


def _load_akshare_histories(
    ak_module,
    tickers: list[str],
    start: str,
    end: str,
    adjust: str,
    cache_dir: Path,
    max_retries: int,
    sleep_seconds: float,
    max_workers: int,
    show_progress: bool,
) -> tuple[list[pd.DataFrame], list[str], list[tuple[str, str]]]:
    if max_workers > 1:
        raise ValueError("AKShare is not thread-safe in this environment; use --max-workers 1.")

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    total = len(tickers)
    for index, ticker in enumerate(tickers, start=1):
        if show_progress:
            print(f"[{index}/{total}] loading {ticker}", flush=True)
        frame, failure, skip_reason = _load_akshare_history_task(
            ak_module, ticker, start, end, adjust, cache_dir, max_retries, sleep_seconds
        )
        if failure:
            failures.append(failure)
        elif skip_reason:
            skipped.append((ticker, skip_reason))
        elif not frame.empty:
            frames.append(frame)
        sleep(sleep_seconds)
    return frames, failures, skipped


def _load_akshare_history_task(
    ak_module,
    ticker: str,
    start: str,
    end: str,
    adjust: str,
    cache_dir: Path,
    max_retries: int,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, str | None, str | None]:
    try:
        frame = _load_one_akshare_history_with_retry(
            ak_module=ak_module,
            ticker=ticker,
            start=start,
            end=end,
            adjust=adjust,
            cache_dir=cache_dir,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
        return frame, None, None
    except Exception as exc:  # pragma: no cover - depends on network and upstream schema.
        if "returned empty daily data" in str(exc):
            return pd.DataFrame(), None, "no daily data in backtest window, likely listed after end date"
        return pd.DataFrame(), f"{ticker}: {exc}", None


def add_adjusted_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Create adjusted OHLC prices when an adjustment factor is available."""

    result = frame.copy()
    if "adj_factor" not in result.columns:
        result["adj_factor"] = 1.0

    for column in ["open", "high", "low", "close"]:
        if column in result.columns:
            result[f"adj_{column}"] = result[column] * result["adj_factor"]

    if "adj_close" not in result.columns and "close" in result.columns:
        result["adj_close"] = result["close"] * result["adj_factor"]
    return result


def _load_one_akshare_history(
    ak_module,
    ticker: str,
    start: str,
    end: str,
    adjust: str,
    cache_dir: Path,
) -> pd.DataFrame:
    cache_file = cache_dir / f"{_safe_cache_key(ticker)}_{start}_{end}_{adjust or 'none'}.csv"
    if cache_file.exists():
        try:
            return pd.read_csv(cache_file, parse_dates=["date"])
        except pd.errors.EmptyDataError:
            cache_file.unlink(missing_ok=True)

    raw = _fetch_akshare_daily_raw(ak_module, ticker, start, end, adjust)
    frame = _normalize_akshare_daily(raw, ticker)
    if frame.empty:
        raise ValueError(f"AKShare returned empty daily data for {ticker}")
    frame.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return frame


def _fetch_akshare_daily_raw(ak_module, ticker: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    start_date = start.replace("-", "")
    end_date = end.replace("-", "")
    try:
        return ak_module.stock_zh_a_daily(
            symbol=_to_akshare_prefixed_symbol(ticker),
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    except Exception:
        return ak_module.stock_zh_a_hist(
            symbol=_to_akshare_symbol(ticker),
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )


def _load_one_akshare_history_with_retry(
    ak_module,
    ticker: str,
    start: str,
    end: str,
    adjust: str,
    cache_dir: Path,
    max_retries: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return _load_one_akshare_history(ak_module, ticker, start, end, adjust, cache_dir)
        except Exception as exc:  # pragma: no cover - depends on network and upstream schema.
            last_error = exc
            if attempt < max_retries:
                sleep(sleep_seconds * attempt)
    raise RuntimeError(f"failed after {max_retries} attempts: {last_error}") from last_error


def _normalize_akshare_daily(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    rename = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_change",
        "换手率": "turnover",
    }
    frame = raw.rename(columns=rename).copy()
    required = ["date", "open", "high", "low", "close", "volume", "amount"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"AKShare response missing columns for {ticker}: {missing}")

    optional = [column for column in ["pct_change", "turnover", "outstanding_share"] if column in frame.columns]
    frame = frame[required + optional]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = ticker
    frame["adj_factor"] = 1.0
    if "outstanding_share" in frame.columns:
        frame["market_cap"] = frame["outstanding_share"] * frame["close"]
    frame["is_suspended"] = frame["volume"].fillna(0).eq(0)
    if "pct_change" in frame.columns:
        frame["is_limit_up"] = frame["pct_change"].fillna(0) >= 9.5
        frame["is_limit_down"] = frame["pct_change"].fillna(0) <= -9.5
    else:
        returns = frame["close"].pct_change()
        frame["is_limit_up"] = returns.fillna(0) >= 0.095
        frame["is_limit_down"] = returns.fillna(0) <= -0.095
    return frame


def _to_akshare_symbol(ticker: str) -> str:
    return str(ticker).split(".")[0].zfill(6)


def _to_akshare_prefixed_symbol(ticker: str) -> str:
    code = _to_akshare_symbol(ticker)
    suffix = str(ticker).split(".")[-1].lower()
    prefix = suffix if suffix in {"sh", "sz"} else ("sh" if code.startswith(("5", "6", "9")) else "sz")
    return f"{prefix}{code}"


def _safe_cache_key(ticker: str) -> str:
    return str(ticker).replace(".", "_")


def clean_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter observations unsuitable for a cross-sectional A-share backtest."""

    result = frame.copy()
    tradable = result["adj_close"].notna()

    for flag in ["is_suspended", "is_limit_up", "is_limit_down"]:
        if flag in result.columns:
            tradable &= ~result[flag].fillna(False).astype(bool)

    for column in ["amount", "volume"]:
        if column in result.columns:
            tradable &= result[column].fillna(0) > 0

    return result.loc[tradable].copy()
