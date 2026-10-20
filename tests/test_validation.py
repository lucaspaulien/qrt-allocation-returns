import numpy as np
import pandas as pd

from allocpred.chronology import reconstruct
from allocpred.validation import chronological_folds, purged_train_mask


def test_folds_are_contiguous_blocks_in_every_segment(broken_panel):
    X, _ = broken_panel
    chrono = reconstruct(X)
    fold, seg, rank = chronological_folds(chrono, X.TS.values, n_folds=4)
    assert set(fold) == {0, 1, 2, 3}
    for s in np.unique(seg):
        f = pd.Series(fold[seg == s], index=rank[seg == s]).groupby(level=0).first().sort_index()
        assert (np.diff(f.values) >= 0).all()          # block k precedes block k+1
        assert set(f.values) == {0, 1, 2, 3}


def test_purge_removes_exactly_the_embargo_zone():
    seg = np.array([0] * 100 + [1] * 50)
    rank = np.r_[np.arange(100), np.arange(50)]
    valid = (seg == 0) & (rank >= 40) & (rank < 60)
    train = purged_train_mask(valid, seg, rank, embargo=5)
    assert not (train & valid).any()
    kept = rank[(seg == 0) & train]
    assert kept.max() < 100 and not ((kept >= 35) & (kept < 65)).any()
    assert set(kept) == set(range(35)) | set(range(65, 100))
    assert train[seg == 1].all()                     # other segments are at an unknown distance


def test_purge_handles_several_rows_per_date():
    seg = np.zeros(60, int)
    rank = np.repeat(np.arange(20), 3)
    valid = rank == 10
    train = purged_train_mask(valid, seg, rank, embargo=2)
    assert set(rank[train]) == set(range(8)) | set(range(13, 20))
