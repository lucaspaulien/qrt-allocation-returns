"""Purged, chronological cross-validation on the reconstructed calendar.

A random K-fold on the TS labels puts neighbouring days on both sides of the
split, although their 20-day windows overlap and a date's target reappears in
the features of the next date. With the calendar rebuilt (``chronology.py``),
each segment of consecutive days is cut into ``n_folds`` contiguous blocks,
and any training row within ``embargo`` days of a validation row is dropped.
"""
import numpy as np
import pandas as pd


def chronological_folds(chrono, ts, n_folds=5):
    """Fold id of every row, plus its (segment, rank) position in the calendar.

    ``chrono`` has columns TS, segment, rank; ``ts`` is the TS label of each row.
    Block k of every segment goes to fold k, so each fold spans all segments.
    """
    pos = pd.DataFrame({'TS': np.asarray(ts)}).merge(chrono, on='TS', how='left')
    if pos.segment.isna().any():
        raise ValueError('some rows have a date that is not in the chronology')
    seg, rank = pos.segment.to_numpy(int), pos['rank'].to_numpy(int)
    fold_of = {}
    for s, g in chrono.groupby('segment'):
        for k, block in enumerate(np.array_split(np.sort(g['rank'].values), n_folds)):
            for p in block:
                fold_of[(s, p)] = k
    fold = np.array([fold_of[(s, p)] for s, p in zip(seg, rank)])
    return fold, seg, rank


def purged_train_mask(valid, seg, rank, embargo=21):
    """Training rows for a validation mask: everything else, minus the embargo zone.

    A row is purged when it sits in the same segment as some validation row and
    at most ``embargo`` days away from it. 21 days covers the 20-day window plus
    the one-day-ahead target.
    """
    near = np.zeros(len(valid), bool)
    by_segment = {}
    for s, p in zip(seg[valid], rank[valid]):
        by_segment.setdefault(s, []).append(p)
    for s, ranks in by_segment.items():
        ranks = np.unique(ranks)
        here = np.where(seg == s)[0]
        i = np.searchsorted(ranks, rank[here])
        before = ranks[np.clip(i - 1, 0, len(ranks) - 1)]
        after = ranks[np.clip(i, 0, len(ranks) - 1)]
        d = np.minimum(np.abs(rank[here] - before), np.abs(rank[here] - after))
        near[here[d <= embargo]] = True
    return ~valid & ~near
