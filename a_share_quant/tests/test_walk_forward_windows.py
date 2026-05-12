from __future__ import annotations

import pandas as pd

from ashare_quant.walk_forward import WalkForwardFold, fold_test_mask, fold_train_mask, walk_forward_folds


def test_walk_forward_half_open_no_train_test_overlap() -> None:
    folds = walk_forward_folds(
        "2021-01-01",
        "2024-12-31",
        train_months=24,
        test_months=6,
        step_months=6,
    )
    assert len(folds) >= 1
    for f in folds:
        assert f.train_start < f.train_end_exclusive
        assert f.test_start == f.train_end_exclusive
        assert f.test_start < f.test_end_exclusive
        assert f.train_end_exclusive <= f.test_start


def test_walk_forward_test_segments_non_overlapping_when_step_equals_test() -> None:
    folds = walk_forward_folds("2021-01-01", "2024-06-30", train_months=12, test_months=3, step_months=3)
    tests = [(f.test_start, f.test_end_exclusive) for f in folds]
    for i in range(len(tests) - 1):
        _, end_i = tests[i]
        start_j, _ = tests[i + 1]
        assert end_i <= start_j, "OOS test windows must not overlap when step == test length"


def test_fold_masks_monotone_dates() -> None:
    f = WalkForwardFold(
        train_start=pd.Timestamp("2021-01-01"),
        train_end_exclusive=pd.Timestamp("2022-01-01"),
        test_start=pd.Timestamp("2022-01-01"),
        test_end_exclusive=pd.Timestamp("2022-07-01"),
    )
    dates = pd.date_range("2020-06-01", "2023-01-01", freq="B")
    tm = fold_train_mask(dates, f)
    xm = fold_test_mask(dates, f)
    assert not (tm & xm).any()
    assert tm.sum() > 0 and xm.sum() > 0
