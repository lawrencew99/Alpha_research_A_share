"""Date-range slicing for train / validation / test research workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

EvalWindow = Literal["train", "val", "test", "train_val", "all"]


@dataclass(frozen=True)
class SampleSplitConfig:
    """Calendar split for factor fit, hyper-parameter search, and hold-out test."""

    train_start: str
    train_end: str
    val_start: str | None = None
    val_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    @staticmethod
    def research_defaults_hs300() -> SampleSplitConfig:
        return SampleSplitConfig(
            train_start="2021-01-01",
            train_end="2022-12-31",
            val_start="2023-01-01",
            val_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-12-31",
        )

    def validate(self) -> None:
        ts = _parse(self.train_start)
        te = _parse(self.train_end)
        if ts > te:
            raise ValueError("train_start must be <= train_end")
        if self.val_start is not None and self.val_end is not None:
            vs, ve = _parse(self.val_start), _parse(self.val_end)
            if vs > ve:
                raise ValueError("val_start must be <= val_end")
            if ve <= te:
                raise ValueError("validation must end strictly after train_end")
        if self.test_start is not None and self.test_end is not None:
            tts, tte = _parse(self.test_start), _parse(self.test_end)
            if tts > tte:
                raise ValueError("test_start must be <= test_end")
            if self.val_end is not None:
                ve = _parse(self.val_end)
                if tts <= ve:
                    raise ValueError("test must start strictly after val_end when val is set")
            elif tts <= te:
                raise ValueError("test must start strictly after train_end when val is not set")


def _parse(d: str) -> pd.Timestamp:
    return pd.Timestamp(d)


def slice_panel_by_date(
    frame: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    inclusive: tuple[bool, bool] = (True, True),
) -> pd.DataFrame:
    """Slice a MultiIndex (date, ticker, ...) panel on the date level."""

    if "date" not in frame.index.names:
        raise ValueError("frame index must contain level 'date'")
    d = frame.index.get_level_values("date")
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    mask_lo = d >= lo if inclusive[0] else d > lo
    mask_hi = d <= hi if inclusive[1] else d < hi
    return frame.loc[mask_lo & mask_hi]


def union_split_bounds(cfg: SampleSplitConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest and latest calendar dates referenced by the split."""

    parts: list[pd.Timestamp] = [_parse(cfg.train_start), _parse(cfg.train_end)]
    if cfg.val_start and cfg.val_end:
        parts.extend([_parse(cfg.val_start), _parse(cfg.val_end)])
    if cfg.test_start and cfg.test_end:
        parts.extend([_parse(cfg.test_start), _parse(cfg.test_end)])
    return min(parts), max(parts)


def slice_for_eval_window(frame: pd.DataFrame, cfg: SampleSplitConfig, window: EvalWindow) -> pd.DataFrame:
    if window == "train":
        return slice_panel_by_date(frame, cfg.train_start, cfg.train_end)
    if window == "val":
        if cfg.val_start is None or cfg.val_end is None:
            raise ValueError("val window requested but val dates not set")
        return slice_panel_by_date(frame, cfg.val_start, cfg.val_end)
    if window == "test":
        if cfg.test_start is None or cfg.test_end is None:
            raise ValueError("test window requested but test dates not set")
        return slice_panel_by_date(frame, cfg.test_start, cfg.test_end)
    if window == "train_val":
        if cfg.val_start is None or cfg.val_end is None:
            return slice_panel_by_date(frame, cfg.train_start, cfg.train_end)
        hi = _parse(cfg.val_end)
        return slice_panel_by_date(frame, cfg.train_start, hi)
    if window == "all":
        lo, hi = union_split_bounds(cfg)
        return slice_panel_by_date(frame, lo, hi)
    raise ValueError(f"unknown window: {window!r}")
