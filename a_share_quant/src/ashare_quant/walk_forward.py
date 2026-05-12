"""Rolling train / test windows for stitched out-of-sample evaluation.

本模块是仓库主样本外流程（嵌套滚动 walk-forward）的切窗实现，被
`research_pipeline.run_walk_forward_oos` 与 `research_pipeline.run_nested_walk_forward_oos` 使用，
并通过 `examples/run_walk_forward_hs300.py` 暴露给 CLI。各折使用半开区间 ``[start, end_exclusive)``；
当 ``step_months == test_months`` 时各折测试窗互不重叠，可直接按时间拼接成 OOS 序列。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    test_start: pd.Timestamp
    test_end_exclusive: pd.Timestamp


def walk_forward_folds(
    global_start: str | pd.Timestamp,
    global_end: str | pd.Timestamp,
    *,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
) -> list[WalkForwardFold]:
    """
    Non-overlapping OOS test segments when step_months == test_months.

    Train: dates in [train_start, train_end_exclusive)
    Test: dates in [test_start, test_end_exclusive) with test_start == train_end_exclusive
    """

    g0 = pd.Timestamp(global_start)
    g1 = pd.Timestamp(global_end)
    folds: list[WalkForwardFold] = []
    train_start = g0
    while True:
        train_end_excl = train_start + pd.DateOffset(months=train_months)
        test_start = train_end_excl
        test_end_excl = test_start + pd.DateOffset(months=test_months)
        if train_end_excl > g1 + pd.Timedelta(days=1):
            break
        if test_start > g1:
            break
        test_end_clip = min(test_end_excl, g1 + pd.Timedelta(days=1))
        if test_start >= test_end_clip:
            break
        folds.append(
            WalkForwardFold(
                train_start=train_start,
                train_end_exclusive=train_end_excl,
                test_start=test_start,
                test_end_exclusive=test_end_clip,
            )
        )
        train_start += pd.DateOffset(months=step_months)
        if train_start > g1:
            break
    return folds


def fold_train_mask(dates: pd.DatetimeIndex, fold: WalkForwardFold) -> pd.Series:
    return (dates >= fold.train_start) & (dates < fold.train_end_exclusive)


def fold_test_mask(dates: pd.DatetimeIndex, fold: WalkForwardFold) -> pd.Series:
    return (dates >= fold.test_start) & (dates < fold.test_end_exclusive)
