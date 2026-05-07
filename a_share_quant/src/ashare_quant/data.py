"""Data loading, adjustment and A-share universe cleaning utilities."""

from __future__ import annotations

from pathlib import Path
from time import sleep

import pandas as pd


REQUIRED_PRICE_COLUMNS = {"date", "ticker", "close"}


def limit_pct_for_ticker(ticker: str) -> float:
    """Return daily limit magnitude in percent points for the board (e.g. 10.0 for ±10%)."""

    raw = str(ticker).strip().upper()
    code = raw.split(".")[0]
    suffix = raw.split(".")[-1] if "." in raw else ""

    if suffix == "BJ" or code.startswith(("8", "4")):
        return 30.0
    if code.startswith(("300", "301")):
        return 20.0
    if code.startswith("688"):
        return 20.0
    return 10.0


def refresh_trade_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute limit-up/down and can_buy/can_sell from price or pct_change (per-ticker board rules)."""

    result = frame.copy()
    idx_names: list | None = None
    had_multiindex = isinstance(result.index, pd.MultiIndex) and result.index.nlevels >= 2
    if had_multiindex:
        idx_names = list(result.index.names)
    if "ticker" not in result.columns:
        result = result.reset_index()

    tickers = result["ticker"].astype(str)
    lim_pct = tickers.map(limit_pct_for_ticker)
    threshold_pct = (lim_pct - 0.05).astype(float)

    if "pct_change" in result.columns:
        pc = pd.to_numeric(result["pct_change"], errors="coerce").fillna(0.0)
        result["is_limit_up"] = pc >= threshold_pct
        result["is_limit_down"] = pc <= -threshold_pct
    elif "close" in result.columns:
        ordered = result.sort_values(["ticker", "date"]) if "date" in result.columns else result
        ret = ordered.groupby("ticker", group_keys=False)["close"].pct_change(fill_method=None)
        thr_dec = ordered["ticker"].map(lambda t: limit_pct_for_ticker(t) / 100.0 - 0.001)
        lu_series = ret.fillna(0.0) >= thr_dec
        ld_series = ret.fillna(0.0) <= -thr_dec
        result["is_limit_up"] = lu_series.reindex(result.index).fillna(False)
        result["is_limit_down"] = ld_series.reindex(result.index).fillna(False)
    else:
        result["is_limit_up"] = False
        result["is_limit_down"] = False

    if "is_suspended" not in result.columns:
        if "volume" in result.columns:
            result["is_suspended"] = result["volume"].fillna(0).eq(0)
        else:
            result["is_suspended"] = False

    sus = result["is_suspended"].fillna(False).astype(bool)
    lu = result["is_limit_up"].fillna(False).astype(bool)
    ld = result["is_limit_down"].fillna(False).astype(bool)
    result["can_buy"] = ~(lu | sus)
    result["can_sell"] = ~(ld | sus)

    if had_multiindex and "date" in result.columns and "ticker" in result.columns:
        result = result.set_index(["date", "ticker"]).sort_index()
        if idx_names and len(idx_names) >= 2 and all(name is not None for name in idx_names):
            result.index.set_names(idx_names, inplace=True)

    return result


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
            cached = pd.read_csv(cache_file, parse_dates=["date"])
            return refresh_trade_flags(cached)
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
    return refresh_trade_flags(frame)


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
    """Filter observations unsuitable for a cross-sectional A-share backtest.

    Keeps limit-up/limit-down rows so daily returns are not lost; use ``can_buy`` /
    ``can_sell`` in execution instead of dropping those bars.
    """

    result = refresh_trade_flags(frame)
    tradable = result["adj_close"].notna()

    if "is_suspended" in result.columns:
        tradable &= ~result["is_suspended"].fillna(False).astype(bool)

    for column in ["amount", "volume"]:
        if column in result.columns:
            tradable &= result[column].fillna(0) > 0

    return result.loc[tradable].copy()


def normalize_benchmark_ticker(benchmark: str) -> str:
    """Map common CSI 300 aliases to the synthetic ticker used in the panel."""

    b = str(benchmark).strip().upper().replace("-", "")
    if b in {"000300", "CSI300", "HS300"}:
        return "000300.SH"
    if b in {"SH000300", "000300.SH"}:
        return "000300.SH"
    return str(benchmark).strip()


def load_akshare_index_history(
    symbol: str,
    out_ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path = "data/raw/akshare_index",
) -> pd.DataFrame:
    """Download index daily OHLCV via AKShare and return a ``(date, ticker)`` panel rowset."""

    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise ImportError('AKShare is required for index history: pip install "ashare-quant[data]"') from exc

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    safe_sym = str(symbol).replace(".", "_")
    cache_file = cache_path / f"index_{safe_sym}_{start}_{end}.csv"

    if cache_file.exists():
        flat = pd.read_csv(cache_file, parse_dates=["date"])
    else:
        start_d = start.replace("-", "")
        end_d = end.replace("-", "")
        raw: pd.DataFrame | None = None
        last_err: Exception | None = None
        try:
            raw = ak.index_zh_a_hist(symbol=str(symbol).zfill(6), period="daily", start_date=start_d, end_date=end_d)
        except Exception as exc:  # pragma: no cover - network/schema
            last_err = exc
        if raw is None or raw.empty:
            try:
                sym = f"sh{str(symbol).zfill(6)}"
                alt = ak.stock_zh_index_daily(symbol=sym)
                if alt is not None and not alt.empty:
                    alt = alt.copy()
                    alt["date"] = pd.to_datetime(alt["date"])
                    raw = alt.loc[(alt["date"] >= pd.Timestamp(start)) & (alt["date"] <= pd.Timestamp(end))]
            except Exception as exc:  # pragma: no cover
                last_err = exc
        if raw is None or raw.empty:
            raise RuntimeError(f"failed to load index {symbol}: {last_err}")
        flat = _normalize_index_hist_raw(raw, out_ticker)
        flat.to_csv(cache_file, index=False, encoding="utf-8-sig")

    flat = refresh_trade_flags(flat)
    panel = flat.set_index(["date", "ticker"]).sort_index()
    return add_adjusted_prices(panel)


def _normalize_index_hist_raw(raw: pd.DataFrame, out_ticker: str) -> pd.DataFrame:
    rename_candidates = {
        "日期": "date",
        "date": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_change",
    }
    rename_map = {k: v for k, v in rename_candidates.items() if k in raw.columns}
    frame = raw.rename(columns=rename_map).copy()
    if "date" not in frame.columns:
        frame["date"] = pd.to_datetime(raw.iloc[:, 0])

    for column in ["open", "high", "low", "close"]:
        if column not in frame.columns and "close" in frame.columns:
            frame[column] = frame["close"]

    if "volume" not in frame.columns:
        frame["volume"] = 1e12
    if "amount" not in frame.columns:
        frame["amount"] = pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(
            frame["volume"], errors="coerce"
        ).replace(0, 1)

    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(1e12).clip(lower=1.0)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(
        pd.to_numeric(frame["close"], errors="coerce") * frame["volume"]
    )

    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = out_ticker
    frame["adj_factor"] = 1.0
    keep = ["date", "ticker", "open", "high", "low", "close", "volume", "amount"]
    if "pct_change" in frame.columns:
        keep.append("pct_change")
    return frame[keep]


def merge_benchmark_if_needed(
    market: pd.DataFrame,
    benchmark: str | None,
    start: str,
    end: str,
    cache_dir: str | Path = "data/raw/akshare_index",
) -> pd.DataFrame:
    """Append a synthetic index ticker column when ``benchmark`` names an index not in ``market``."""

    if benchmark is None:
        return market
    key = str(benchmark).strip().lower()
    if key in {"", "none", "equal_weight", "universe_equal"}:
        return market

    target = normalize_benchmark_ticker(str(benchmark))
    existing = market.index.get_level_values("ticker").unique()
    if target in existing:
        return market

    code = target.split(".")[0].zfill(6)
    idx_panel = load_akshare_index_history(
        symbol=code,
        out_ticker=target,
        start=start,
        end=end,
        cache_dir=cache_dir,
    )
    return _concat_align_market_frames(market, idx_panel)


def _concat_align_market_frames(stocks: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two panels with identical columns."""

    extra = extra.copy()
    for column in stocks.columns:
        if column not in extra.columns:
            if column == "turnover":
                extra[column] = 0.0
            elif column in {"outstanding_share", "market_cap"}:
                extra[column] = float("nan")
            elif column in {"is_suspended", "is_limit_up", "is_limit_down"}:
                extra[column] = False
            elif column in {"can_buy", "can_sell"}:
                extra[column] = True
            else:
                extra[column] = float("nan")
    extra = extra.reindex(columns=stocks.columns)
    return pd.concat([stocks, extra]).sort_index()
