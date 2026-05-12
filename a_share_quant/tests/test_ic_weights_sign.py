from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_quant.ic_weights import estimate_factor_weights_ic


def _synth_panel(
    dates: pd.DatetimeIndex,
    tickers: list[str],
    forward: pd.Series,
    pos_strength: float,
    neg_strength: float,
) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    fr = forward.reindex(idx)
    rng = np.random.default_rng(0)
    pos = rng.standard_normal(len(idx))
    neg = rng.standard_normal(len(idx))
    factors = pd.DataFrame(
        {
            "pos": pos * 0.01 + fr.groupby(level="date").transform(lambda s: s.fillna(0)) * pos_strength,
            "neg": neg * 0.01 - fr.groupby(level="date").transform(lambda s: s.fillna(0)) * neg_strength,
        },
        index=idx,
    )
    return factors, fr


def test_ic_weights_positive_for_positive_ic_factor() -> None:
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    tickers = [f"{i:06d}.X" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    fwd = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx, name="fwd")
    factors, fr = _synth_panel(dates, tickers, fwd, pos_strength=3.0, neg_strength=3.0)
    w = estimate_factor_weights_ic(factors, fr, mode="ic_mean", clip_abs=10.0, min_ic_dates=10)
    assert w["pos"] > 0
    assert w["neg"] < 0
