"""A synthetic panel built exactly like the challenge data, with a known calendar."""
import numpy as np
import pandas as pd
import pytest

from allocpred.data import RET, SIGNED_VOLUME


def make_panel(n_days=160, n_alloc=40, seed=0, breaks=(), noise=1e-4, missing=0.1, phi=None):
    """Rows (date, allocation) with 20-day return windows over a latent return path.

    Row of day t: RET_k = r[t - k] (+ noise), target = r[t], so the target of t
    is the RET_1 of t + 1. ``breaks`` are day indices after which 30 days are
    skipped, creating separate calendar segments. Date labels are shuffled.
    Allocations get an AR(1) coefficient ``phi`` (momentum loading).
    """
    rng = np.random.RandomState(seed)
    days = np.arange(n_days)
    for b in breaks:
        days = np.where(days > b, days + 30, days)
    T = days.max() + 22
    phi = rng.uniform(0.0, 0.4, n_alloc) if phi is None else np.full(n_alloc, phi)
    common = rng.normal(0, 0.006, T)
    r = np.zeros((T, n_alloc))
    for t in range(1, T):
        r[t] = phi * r[t - 1] + 0.5 * common[t] + rng.normal(0, 0.01, n_alloc)
    labels = rng.permutation(n_days)
    rows, y = [], []
    for i, d in enumerate(days):
        t = d + 20
        for a in range(n_alloc):
            if rng.rand() < missing:
                continue
            win = r[t - np.arange(1, 21), a] + rng.normal(0, noise, 20)
            sv = rng.normal(0, 1, 20)
            sv[0] = np.nan if rng.rand() < 0.7 else sv[0]
            rows.append(dict(TS=f'DATE_{labels[i] + 1:04d}', ALLOCATION=f'ALLOCATION_{a:02d}',
                             **dict(zip(RET, win)), **dict(zip(SIGNED_VOLUME, sv)),
                             MEDIAN_DAILY_TURNOVER=rng.uniform(0.1, 1), GROUP=a % 4,
                             true_day=d))
            y.append(r[t, a])
    X = pd.DataFrame(rows)
    X.index.name = 'ROW_ID'
    return X, pd.Series(y, index=X.index, name='target')


@pytest.fixture(scope='session')
def panel():
    return make_panel()


@pytest.fixture(scope='session')
def broken_panel():
    return make_panel(breaks=(59, 109), seed=1)
