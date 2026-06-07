"""Time-based walk-forward splitting. Never shuffles — that would leak the future.

``walk_forward_folds`` yields ``Fold`` objects whose train block always precedes
the validation block in time, with no overlap. Fold boundaries come from
``config.SPLITS`` (expanding window across 2015).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import pandas as pd

from rossmann import config

_DATE = config.COLS.date


@dataclass
class Fold:
    index: int
    train: pd.DataFrame
    val: pd.DataFrame

    @property
    def label(self) -> str:
        return (
            f"Fold {self.index}: "
            f"train <= {self.train[_DATE].max().date()} "
            f"| val {self.val[_DATE].min().date()} -> {self.val[_DATE].max().date()}"
        )


def walk_forward_folds(df: pd.DataFrame, splits: List[tuple] | None = None) -> Iterator[Fold]:
    """Yield expanding-window train/validation folds defined by ``splits``.

    Each split is ``(train_end, val_start, val_end)`` as ISO date strings.
    """
    splits = splits or config.SPLITS.folds
    for i, (train_end, val_start, val_end) in enumerate(splits, start=1):
        train_end = pd.Timestamp(train_end)
        val_start = pd.Timestamp(val_start)
        val_end = pd.Timestamp(val_end)

        train = df[df[_DATE] <= train_end]
        val = df[(df[_DATE] >= val_start) & (df[_DATE] <= val_end)]
        if len(train) == 0 or len(val) == 0:
            continue
        yield Fold(index=i, train=train.copy(), val=val.copy())


def holdout_split(df: pd.DataFrame, val_weeks: int = 6) -> Fold:
    """Single hold-out: last ``val_weeks`` weeks as validation, rest as train."""
    cutoff = df[_DATE].max() - pd.Timedelta(weeks=val_weeks)
    train = df[df[_DATE] <= cutoff]
    val = df[df[_DATE] > cutoff]
    return Fold(index=0, train=train.copy(), val=val.copy())
