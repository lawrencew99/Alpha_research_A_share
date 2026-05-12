"""Factor construction, cross-sectional normalization and neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_quant.config import FactorConfig


def build_factor_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Build common A-share alpha factors from a cleaned market panel."""

    result = pd.DataFrame(index=frame.index)
    close = frame["adj_close"]
    by_ticker = close.groupby(level="ticker", group_keys=False)
    daily_return = by_ticker.pct_change()

    result["momentum_60"] = by_ticker.pct_change(60)
    result["reversal_20"] = -by_ticker.pct_change(20)
    result["volatility_20"] = daily_return.groupby(level="ticker", group_keys=False).transform(
        lambda values: values.rolling(20).std()
    )

    if "amount" in frame.columns:
        amount = frame["amount"].replace(0, np.nan)
        result["liquidity_20"] = np.log(amount).groupby(level="ticker", group_keys=False).transform(
            lambda values: values.rolling(20).mean()
        )

    if "pb" in frame.columns:
        result["value_pb"] = -np.log(frame["pb"].replace(0, np.nan))
    if "roe" in frame.columns:
        result["quality_roe"] = frame["roe"]

    context_columns = [column for column in ["industry", "market_cap"] if column in frame.columns]
    return result.join(frame[context_columns])


def combine_factors(
    factors: pd.DataFrame,
    config: FactorConfig | None = None,
    *,
    ic_calibration_factors: pd.DataFrame | None = None,
    ic_calibration_future: pd.Series | None = None,
) -> pd.Series:
    """Create a composite multi-factor score after cleaning and neutralization."""

    config = config or FactorConfig()
    if config.weights_source == "ic_train":
        if ic_calibration_factors is None or ic_calibration_future is None:
            raise ValueError("ic_train requires ic_calibration_factors and ic_calibration_future")
        from ashare_quant.ic_weights import estimate_factor_weights_ic

        weight_map = estimate_factor_weights_ic(ic_calibration_factors, ic_calibration_future)
        fw = {k: float(v) for k, v in weight_map.items() if k in factors.columns and abs(float(v)) > 1e-12}
        if not fw:
            raise ValueError("IC weight estimation returned no usable factors")
        config = FactorConfig(
            winsor_quantile=config.winsor_quantile,
            neutralize_industry=config.neutralize_industry,
            neutralize_size=config.neutralize_size,
            weights_source="manual",
            factor_weights=fw,
        )

    factor_columns = [
        column
        for column in config.factor_weights
        if column in factors.columns and abs(float(config.factor_weights[column])) > 1e-12
    ]
    if not factor_columns:
        raise ValueError("no configured factor columns were found")

    processed = pd.DataFrame(index=factors.index)
    for column in factor_columns:
        series = winsorize_by_date(factors[column], config.winsor_quantile)
        series = zscore_by_date(series)
        if config.neutralize_industry or config.neutralize_size:
            series = neutralize_by_date(
                series,
                industry=factors.get("industry") if config.neutralize_industry else None,
                market_cap=factors.get("market_cap") if config.neutralize_size else None,
            )
            series = zscore_by_date(series)
        processed[column] = series

    weights = pd.Series(config.factor_weights, dtype=float).reindex(factor_columns)
    weights = weights / weights.abs().sum()
    score = processed.mul(weights, axis=1).sum(axis=1, min_count=1)
    score.name = "score"
    return zscore_by_date(score)


def winsorize_by_date(series: pd.Series, quantile: float = 0.025) -> pd.Series:
    """Clip each date's cross-section to reduce extreme value impact."""

    def _clip(cross_section: pd.Series) -> pd.Series:
        low, high = cross_section.quantile([quantile, 1 - quantile])
        return cross_section.clip(low, high)

    return series.groupby(level="date", group_keys=False).apply(_clip)


def zscore_by_date(series: pd.Series) -> pd.Series:
    """Standardize each date's cross-section."""

    def _zscore(cross_section: pd.Series) -> pd.Series:
        std = cross_section.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return cross_section * np.nan
        return (cross_section - cross_section.mean()) / std

    return series.groupby(level="date", group_keys=False).apply(_zscore)


def neutralize_by_date(
    series: pd.Series,
    industry: pd.Series | None = None,
    market_cap: pd.Series | None = None,
) -> pd.Series:
    """Regress out industry dummies and log market-cap exposure by date."""

    pieces: list[pd.Series] = []
    for date, y in series.groupby(level="date"):
        y_cross = y.droplevel("date")
        residual = _neutralize_cross_section(
            y_cross,
            industry=_slice_date(industry, date),
            market_cap=_slice_date(market_cap, date),
        )
        residual.index = pd.MultiIndex.from_product(
            [[date], residual.index],
            names=series.index.names,
        )
        pieces.append(residual)
    return pd.concat(pieces).sort_index()


def _neutralize_cross_section(
    y: pd.Series,
    industry: pd.Series | None,
    market_cap: pd.Series | None,
) -> pd.Series:
    data = pd.DataFrame({"factor": y})
    if industry is not None:
        data["industry"] = industry.astype("category")
    if market_cap is not None:
        data["log_size"] = np.log(market_cap.replace(0, np.nan))

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    residual = pd.Series(np.nan, index=y.index, name=y.name)
    if len(data) < 5:
        return residual

    x_parts = [pd.Series(1.0, index=data.index, name="const")]
    if "log_size" in data:
        x_parts.append(data["log_size"])
    if "industry" in data:
        dummies = pd.get_dummies(data["industry"], drop_first=True, dtype=float)
        if not dummies.empty:
            x_parts.append(dummies)

    x = pd.concat(x_parts, axis=1)
    if len(data) <= x.shape[1] + 1:
        residual.loc[data.index] = data["factor"] - data["factor"].mean()
        return residual

    beta, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), data["factor"].to_numpy(dtype=float), rcond=None)
    fitted = x.to_numpy(dtype=float) @ beta
    residual.loc[data.index] = data["factor"] - fitted
    return residual


def _slice_date(series: pd.Series | None, date: pd.Timestamp) -> pd.Series | None:
    if series is None:
        return None
    return series.xs(date, level="date")
