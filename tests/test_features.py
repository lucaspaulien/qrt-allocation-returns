import numpy as np
import pandas as pd

from allocpred.allocation import CommonFactors, add_loadings, allocation_stats
from allocpred.features import build_features


def test_feature_columns_identical_on_any_split(panel):
    X, _ = panel
    a, b = build_features(X.iloc[:2000]), build_features(X.iloc[2000:])
    assert list(a.columns) == list(b.columns) and len(a.columns) == 28
    assert a.index.equals(X.index[:2000])


def test_cross_sectional_rank_and_market_mean(panel):
    X, _ = panel
    F = build_features(X)
    assert F.RET_1_rk.between(0, 1, inclusive='right').all()
    day_mean = X.groupby('TS').RET_1.transform('mean')
    assert np.allclose(F.RET_1_mkt, day_mean)


def test_group_mean_leaves_the_row_out():
    X = pd.DataFrame({'TS': ['D1'] * 3 + ['D2'], 'ALLOCATION': list('abca'), 'GROUP': [0, 0, 1, 0],
                      'MEDIAN_DAILY_TURNOVER': 1.0})
    for i in range(1, 21):
        X[f'RET_{i}'] = 0.0
        X[f'SIGNED_VOLUME_{i}'] = 0.0
    X['RET_1'] = [1.0, 3.0, 5.0, 7.0]
    F = build_features(X)
    assert F.grp_loo_1.tolist() == [3.0, 1.0, 0.0, 0.0]   # alone in its group -> 0


def test_loadings_only_use_the_training_mask(panel):
    X, y = panel
    alloc, group, ret1 = X.ALLOCATION.values, X.GROUP.values, X.RET_1.values
    mask = np.arange(len(X)) < len(X) // 2
    a = allocation_stats(mask, alloc, group, ret1, y.values)
    y2 = y.values.copy()
    y2[~mask] = np.random.RandomState(0).normal(size=(~mask).sum())
    b = allocation_stats(mask, alloc, group, ret1, y2)
    assert np.allclose(a['A'][0], b['A'][0]) and np.allclose(a['A'][1], b['A'][1])


def test_loadings_recover_momentum_and_shrink(panel):
    X, y = panel
    alloc, group, ret1 = X.ALLOCATION.values, X.GROUP.values, X.RET_1.values
    full = np.ones(len(X), bool)
    weak = allocation_stats(full, alloc, group, ret1, y.values, k=0)
    strong = allocation_stats(full, alloc, group, ret1, y.values, k=1e9)
    # no shrinkage: per-allocation slopes are spread out; infinite shrinkage: all equal the global slope
    assert weak['A'][0].std() > 0.05
    assert np.allclose(strong['A'][0], strong['global'][0])
    F = add_loadings(pd.DataFrame(index=X.index), weak, alloc, group, ret1)
    assert np.allclose(F.mom_charge, F.beta_alloc * ret1)


def test_common_factors(panel):
    X, _ = panel
    cf = CommonFactors(X.TS.values, X.ALLOCATION.values, X.RET_1.values, n_factors=3)
    cf.fit(np.unique(X.TS.values))
    assert np.allclose(cf.L @ cf.L.T, np.eye(3), atol=1e-8)
    F = cf.transform(pd.DataFrame(index=X.index), X.ALLOCATION.values, X.TS.values)
    assert F.shape == (len(X), 3) and np.isfinite(F.values).all()
