"""Per-allocation momentum loadings and cross-sectional common factors.

Loadings use the target, so they are estimated on a training mask only and
must be refitted inside every fold. The factor model only uses explanatory
variables but is refitted per fold too, for consistency.
"""
import numpy as np
import pandas as pd

SHRINKAGE = 800  # pseudo-observations pulling each estimate towards the global value
N_FACTORS = 6


def allocation_stats(mask, alloc, group, ret1, y, k=SHRINKAGE):
    """Shrunk momentum slope and positive rate per allocation (and per group).

    The slope of the target on RET_1 ranges from -0.03 to +0.26 across
    allocations and is persistent (first-half vs second-half estimates
    correlate at 0.41), so it is worth learning. Each estimate is a weighted
    average of the allocation's own value (weight n) and the global value
    (weight ``k``), so rarely seen allocations stay close to the global one.
    """
    d = pd.DataFrame({'A': alloc[mask], 'G': group[mask], 'r': ret1[mask], 'y': y[mask]})
    beta_all = np.cov(d.r, d.y)[0, 1] / d.r.var()
    pos_all = (d.y > 0).mean()
    out = {}
    for col in ['A', 'G']:
        gb = d.groupby(col)
        cov = gb.apply(lambda x: np.cov(x.r, x.y)[0, 1] if len(x) > 5 else np.nan, include_groups=False)
        n = gb.size()
        beta = ((cov / gb.r.var()).fillna(beta_all) * n + k * beta_all) / (n + k)
        pos = (gb.y.apply(lambda z: (z > 0).mean()) * n + k * pos_all) / (n + k)
        out[col] = (beta, pos)
    out['global'] = (beta_all, pos_all)
    return out


def add_loadings(F, stats, alloc, group, ret1):
    bA, pA = stats['A']
    bG, _ = stats['G']
    b0, p0 = stats['global']
    F = F.copy()
    F['beta_alloc'] = pd.Series(alloc).map(bA).fillna(b0).values
    F['pos_alloc'] = pd.Series(alloc).map(pA).fillna(p0).values
    F['beta_grp'] = pd.Series(group).map(bG).fillna(b0).values
    F['mom_charge'] = F.beta_alloc.values * ret1
    F['mom_charge_g'] = F.beta_grp.values * ret1
    return F


def _nan_col_mean(M):
    ok = ~np.isnan(M)
    return np.where(ok.any(0), np.nansum(np.where(ok, M, 0), 0) / np.maximum(ok.sum(0), 1), 0.0)


class CommonFactors:
    """PCA of the date x allocation matrix of RET_1.

    RET_1_mkt only captures the day's average move (the first factor). The PCA
    gives each allocation a loading on ``n_factors`` common sources of risk and
    each date a score; their product per factor is a feature. Built on train
    and test dates together, since no target is involved.
    """

    def __init__(self, ts, alloc, ret1, n_factors=N_FACTORS):
        self.n_factors = n_factors
        self.pivot = (pd.DataFrame({'TS': ts, 'A': alloc, 'r': ret1})
                      .pivot_table(index='TS', columns='A', values='r'))

    def fit(self, dates):
        M = self.pivot.loc[self.pivot.index.isin(dates)].to_numpy(float)
        self.mu = _nan_col_mean(M)
        M = np.nan_to_num(M - self.mu)
        self.sd = M.std(0)
        self.sd[self.sd == 0] = 1
        _, _, Vt = np.linalg.svd(M / self.sd, full_matrices=False)
        self.L = Vt[:self.n_factors]
        return self

    def transform(self, F, alloc, ts):
        P = self.pivot.reindex(index=pd.unique(ts))
        Z = np.nan_to_num((P.to_numpy(float) - self.mu) / self.sd)
        scores = pd.DataFrame(Z @ self.L.T, index=P.index).reindex(ts).to_numpy()
        loads = pd.DataFrame(self.L.T, index=self.pivot.columns).reindex(alloc).to_numpy()
        F = F.copy()
        for j in range(self.n_factors):
            F[f'factor{j}'] = scores[:, j] * loads[:, j]
        return F
