"""Model fitting: a ridge and three LightGBM heads, blended on z-scored outputs.

Three LightGBM objectives look at the same features differently: ``mse`` fits
the raw return, ``huber`` caps the influence of the largest days, ``binary``
fits the sign that the metric scores. Each is averaged over three seeds. The
blend adds the four standardised predictions; the predicted class is its sign.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .allocation import CommonFactors, add_loadings, allocation_stats
from .features import build_features
from .validation import purged_train_mask

PARAMS = dict(verbosity=-1, learning_rate=0.02, max_depth=4, num_leaves=16,
              min_data_in_leaf=3000, lambda_l2=20, feature_fraction=0.5,
              bagging_fraction=0.7, bagging_freq=1, num_threads=4)
# name -> (extra params, label kind, number of trees)
HEADS = {
    'mse': (dict(objective='mse'), 'return', 600),
    'huber': (dict(objective='huber', alpha=0.002), 'return', 900),
    'binary': (dict(objective='binary'), 'sign', 600),
}
SEEDS = (1, 2, 3)
RIDGE_ALPHA = 1000


def zscore(v):
    return (v - v.mean()) / v.std()


class Data:
    """Everything the models need, computed once from the raw frames."""

    def __init__(self, X_train, y_train, X_test):
        self.F_train = build_features(X_train)
        self.F_test = build_features(X_test)
        self.y = y_train.values
        self.sign = (self.y > 0).astype(float)
        self.alloc, self.alloc_test = X_train.ALLOCATION.values, X_test.ALLOCATION.values
        self.group, self.group_test = X_train.GROUP.values, X_test.GROUP.values
        self.ret1, self.ret1_test = self.F_train.RET_1.values, self.F_test.RET_1.values
        self.ts, self.ts_test = X_train.TS.values, X_test.TS.values
        self.factors = CommonFactors(np.r_[self.ts, self.ts_test], np.r_[self.alloc, self.alloc_test],
                                     np.r_[self.ret1, self.ret1_test])

    def label(self, kind):
        return self.y if kind == 'return' else self.sign

    def design(self, mask, test=False):
        """Train (or test) design matrix with loadings and factors fitted on ``mask`` only."""
        stats = allocation_stats(mask, self.alloc, self.group, self.ret1, self.y)
        fac = self.factors.fit(np.unique(self.ts[mask]))
        if test:
            F = add_loadings(self.F_test, stats, self.alloc_test, self.group_test, self.ret1_test)
            return fac.transform(F, self.alloc_test, self.ts_test)
        F = add_loadings(self.F_train, stats, self.alloc, self.group, self.ret1)
        return fac.transform(F, self.alloc, self.ts)


def _ridge(Xa, train, Xp, y):
    M = np.nan_to_num(Xa.drop(columns='GROUP').to_numpy(float))
    P = np.nan_to_num(Xp.drop(columns='GROUP').to_numpy(float))
    sc = StandardScaler().fit(M[train])
    return Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(M[train]), y[train]).predict(sc.transform(P))


def _lgbm(name, Xa, train, Xp, label, seeds=SEEDS, num_threads=None):
    import lightgbm as lgb
    extra, _, n_trees = HEADS[name]
    params = dict(PARAMS, **extra)
    if num_threads:
        params['num_threads'] = num_threads
    acc, models = np.zeros(len(Xp)), []
    for s in seeds:
        m = lgb.train(dict(params, seed=s), lgb.Dataset(Xa[train], label=label[train]),
                      num_boost_round=n_trees)
        acc += m.predict(Xp)
        models.append(m)
    return acc / len(seeds), models


def cross_validate(data, fold, seg, rank, embargo=21, heads=tuple(HEADS), seeds=SEEDS,
                   num_threads=None, log=print):
    """Out-of-fold predictions of the ridge and every LightGBM head.

    Returns (oof, importances): oof maps head name -> prediction for every
    training row; importances holds the gain importances of fold 0's models.
    """
    n = len(data.y)
    oof = {h: np.zeros(n) for h in ('ridge',) + tuple(heads)}
    importances = {}
    for k in np.unique(fold):
        valid = fold == k
        train = purged_train_mask(valid, seg, rank, embargo)
        Xa = data.design(train)
        Xv = Xa[valid]
        oof['ridge'][valid] = _ridge(Xa, train, Xv, data.y)
        for h in heads:
            oof[h][valid], models = _lgbm(h, Xa, train, Xv, data.label(HEADS[h][1]), seeds, num_threads)
            if k == 0:
                importances[h] = np.array([m.feature_importance('gain') for m in models])
        log(f'fold {k}: valid {valid.sum()} rows, train {train.sum()}, purged {(~valid).sum() - train.sum()}')
    importances['columns'] = list(Xa.columns)
    return oof, importances


def blend(preds):
    return sum(zscore(p) for p in preds.values())


def accuracy(oof, sign):
    """Sign accuracy of every head and of the blend (binary head thresholded at 0.5)."""
    out = {h: accuracy_score(sign, (p > (0.5 if h == 'binary' else 0.0)).astype(int)) for h, p in oof.items()}
    out['blend'] = accuracy_score(sign, (blend(oof) > 0).astype(int))
    return out


def fit_predict_test(data, heads=tuple(HEADS), seeds=SEEDS, num_threads=None):
    """Refit everything on the whole training set and return the test blend score."""
    full = np.ones(len(data.y), bool)
    Xa, Xt = data.design(full), data.design(full, test=True)
    preds = {'ridge': _ridge(Xa, full, Xt, data.y)}
    for h in heads:
        preds[h], _ = _lgbm(h, Xa, full, Xt, data.label(HEADS[h][1]), seeds, num_threads)
    return blend(preds)
