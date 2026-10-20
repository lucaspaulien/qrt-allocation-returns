"""Target-free row features (computed identically on train and test).

Everything here only uses the explanatory columns of the row itself or of the
cross-section of the same date. Each test date comes with its full
cross-section, so the date-level aggregates are available on both sides.
Features that use the target (per-allocation loadings) live in
``allocation.py`` because they must be refitted inside each fold.
"""
import numpy as np
import pandas as pd

from .data import RET, SIGNED_VOLUME


def build_features(df):
    r = df[RET].to_numpy(float)
    sv = df[SIGNED_VOLUME].to_numpy(float)
    F = pd.DataFrame(index=df.index)

    # raw momentum: almost all the signal sits in RET_1
    for i in range(1, 6):
        F[f'RET_{i}'] = r[:, i - 1]
    for h in [5, 10, 20]:
        F[f'mom{h}'] = np.nanmean(r[:, :h], 1)

    # risk-scaled momentum: the size of RET_1 relative to the allocation's own vol
    F['vol20'] = np.nanstd(r, 1)
    F['vol5'] = np.nanstd(r[:, :5], 1)
    F['zmom1'] = r[:, 0] / F.vol20
    F['zmom5'] = F.mom5 / F.vol20
    F['MDT'] = df.MEDIAN_DAILY_TURNOVER
    F['GROUP'] = df.GROUP
    F['posfrac'] = np.nanmean(r > 0, 1)

    # SIGNED_VOLUME_1 is missing 3 times out of 4 (the last day is not consolidated
    # yet), so volume features start at lag 2 ...
    F['sv_last'] = sv[:, 1]
    F['crowd_last'] = np.sign(r[:, 1]) * sv[:, 1]
    # ... but whether it is present is itself informative: P(target > 0) moves
    # from 0.5033 when it is missing to 0.5179 when it is there
    F['sv1_dispo'] = df.SIGNED_VOLUME_1.notna().astype(int).values

    # lag-1 autocorrelation of the 20-day window
    rd = r - np.nanmean(r, 1, keepdims=True)
    a, b = rd[:, :-1], rd[:, 1:]
    ok = ~np.isnan(a) & ~np.isnan(b)
    num = np.nansum(np.where(ok, a * b, 0), 1)
    den = np.sqrt(np.nansum(np.where(ok, a * a, 0), 1) * np.nansum(np.where(ok, b * b, 0), 1))
    F['ac1'] = num / np.maximum(den, 1e-12)

    # position in the day's cross-section
    F['TS'] = df.TS.values
    g = F.groupby('TS')
    F['RET_1_rk'] = g.RET_1.rank(pct=True)
    F['mom5_rk'] = g.mom5.rank(pct=True)
    for c in ['RET_1', 'mom5', 'posfrac']:
        F[c + '_mkt'] = g[c].transform('mean')

    # beta to the day's average allocation over the window, and the residual of RET_1
    mkt = (pd.DataFrame(r, index=df.index)
           .assign(TS=df.TS.values).groupby('TS').transform('mean').to_numpy())
    mktd = mkt - np.nanmean(mkt, 1, keepdims=True)
    cov = np.nanmean(rd * mktd, 1)
    var = np.nanmean(mktd * mktd, 1)
    F['beta'] = cov / np.maximum(var, 1e-12)
    F['idio_ret1'] = r[:, 0] - F.beta * mkt[:, 0]

    # same-day mean of the allocation's GROUP, leaving the allocation itself out
    base = pd.DataFrame({'TS': df.TS.values, 'GROUP': df.GROUP.values,
                         'r1': r[:, 0], 'm5': F.mom5.values}, index=df.index)
    for src, name in [('r1', 'grp_loo_1'), ('m5', 'grp_loo_mom5')]:
        gb = base.groupby(['TS', 'GROUP'])[src]
        F[name] = ((gb.transform('sum') - base[src].fillna(0))
                   / np.maximum(gb.transform('count') - base[src].notna(), 1))

    return F.drop(columns='TS')
