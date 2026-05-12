"""IC / ICIR-driven factor weights for research workflows."""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

from ashare_quant.analysis import factor_ic, summarize_ic

WeightMappingMode = Literal["icir", "ic_mean"]


def estimate_factor_weights_ic(
    factors: pd.DataFrame,
    forward_returns: pd.Series,
    *,
    mode: WeightMappingMode = "icir",
    clip_abs: float = 3.0,
    annualization: int = 252,
    min_ic_dates: int = 20,
) -> dict[str, float]:
    """
    Map per-factor IC diagnostics to signed weights and L1-normalize to sum(|w|)=1.

    IC / ICIR are computed with ``factor_ic`` + ``summarize_ic`` (Rank IC by default in factor_ic).
    """

    ic = factor_ic(factors, forward_returns)
    if ic.empty or len(ic) < min_ic_dates:
        raise ValueError("insufficient IC history for weight estimation")
    summary = summarize_ic(ic, annualization=annualization)
    raw = summary["icir"] if mode == "icir" else summary["mean"] * np.sqrt(annualization)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty:
        raise ValueError("no usable factors after IC summary")
    clipped = raw.clip(-clip_abs, clip_abs)
    denom = float(np.abs(clipped.to_numpy(dtype=float)).sum())
    if denom <= 0:
        raise ValueError("IC weights degenerate: zero L1 mass")
    weights = (clipped / denom).to_dict()
    return {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}


def factor_config_weights_from_ic(
    base_factor_columns: tuple[str, ...] | None = None,
    *,
    factors: pd.DataFrame,
    forward_returns: pd.Series,
    mode: WeightMappingMode = "icir",
    clip_abs: float = 3.0,
) -> dict[str, float]:
    """Return a weight dict restricted to typical multi-factor columns when base is given."""

    w = estimate_factor_weights_ic(factors, forward_returns, mode=mode, clip_abs=clip_abs)
    if base_factor_columns is None:
        return w
    return {k: v for k, v in w.items() if k in base_factor_columns}
