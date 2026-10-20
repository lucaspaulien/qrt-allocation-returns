"""End to end on the synthetic panel: reconstruct, fold, fit, predict."""
import numpy as np
import pytest

from allocpred import pipeline as P
from allocpred.chronology import reconstruct
from allocpred.validation import chronological_folds


@pytest.fixture
def small_trees(monkeypatch):
    # the real settings need >= 3000 rows per leaf, far more than the synthetic panel
    monkeypatch.setitem(P.PARAMS, 'min_data_in_leaf', 50)
    monkeypatch.setitem(P.PARAMS, 'num_threads', 2)
    monkeypatch.setattr(P, 'HEADS', {k: (v[0], v[1], 100) for k, v in P.HEADS.items()})


def test_cv_beats_base_rate_on_momentum_panel(small_trees):
    from conftest import make_panel
    X, y = make_panel(n_days=220, n_alloc=40, seed=3, breaks=(120,), phi=0.35)
    X_tr, y_tr = X[X.true_day < 170], y[X.true_day < 170]
    X_te = X[X.true_day >= 170]
    fold, seg, rank = chronological_folds(reconstruct(X_tr), X_tr.TS.values, n_folds=3)
    data = P.Data(X_tr, y_tr, X_te)
    oof, imp = P.cross_validate(data, fold, seg, rank, embargo=5, seeds=(1,), log=lambda s: None)
    acc = P.accuracy(oof, data.sign)
    base = max(data.sign.mean(), 1 - data.sign.mean())
    assert acc['blend'] > base + 0.05
    assert acc['ridge'] > base + 0.05
    assert 'beta_alloc' in imp['columns'] and 'factor0' in imp['columns']

    score = P.fit_predict_test(data, seeds=(1,))
    assert score.shape == (len(X_te),) and np.isfinite(score).all()
