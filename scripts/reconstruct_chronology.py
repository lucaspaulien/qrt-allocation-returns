"""Rebuild the calendar of the shuffled dates and check whether the test set can be chained.

Prints the diagnostics quoted in the README and saves a figure of the
best-vs-second-best successor similarity.

    python scripts/reconstruct_chronology.py --data-dir data --plot docs/chronology.png
"""
import argparse

import numpy as np
from sklearn.metrics import accuracy_score

from allocpred.chronology import (chain_segments, match_successors, overlap_similarity,
                                  return_tensor, target_from_successors)
from allocpred.data import default_data_dir, load


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=str(default_data_dir()))
    ap.add_argument('--plot', default=None, help='save the similarity figure here')
    ap.add_argument('--gaps', type=int, nargs='+', default=[1, 2, 3, 5, 8, 10, 12])
    args = ap.parse_args()

    X_train, y_train, X_test, _ = load(args.data_dir)
    print('naive check: are the labels in order? target(t) vs RET_1 of DATE_{t+1}')
    t = X_train[['TS', 'ALLOCATION', 'RET_1']].assign(tnum=X_train.TS.str.slice(5).astype(int), target=y_train)
    nxt = t.assign(tnum=t.tnum - 1)[['tnum', 'ALLOCATION', 'RET_1']]
    m = t.merge(nxt, on=['tnum', 'ALLOCATION'], suffixes=('', '_next'))
    print(f'  sign agreement {((m.target > 0) == (m.RET_1_next > 0)).mean():.4f}  (0.5 = shuffled)')

    dates, allocs, V, available = return_tensor([X_train, X_test])
    print(f'tensor {V.shape}, {available.mean():.3f} of (date, allocation) pairs present')
    S = overlap_similarity(V, available, gap=1)
    best = S.max(1)
    second = np.partition(S, -2, axis=1)[:, -2]
    print(f'best successor similarity, median {np.median(best):.4f}; best - second, median {np.median(best - second):.4f}')

    succ, quality = match_successors(S)
    is_test = np.isin(dates, X_test.TS.unique())
    for name, sel in [('train', ~is_test), ('test', is_test)]:
        print(f'{name:5s} links: mean quality {quality[sel].mean():.4f}, '
              f'weak (< 0.9) {(quality[sel] < 0.9).sum()} of {sel.sum()}')

    d_idx = X_train.TS.map({d: i for i, d in enumerate(dates)}).values
    a_idx = X_train.ALLOCATION.map({a: i for i, a in enumerate(allocs)}).values
    E = target_from_successors(V, succ, d_idx, a_idx, max_lag=3)
    pos = (y_train > 0).astype(int).values
    for k in range(3):
        ok = ~np.isnan(E[k])
        print(f'target read from lag {k + 1} of the chain: accuracy {accuracy_score(pos[ok], E[k][ok] > 0):.4f}')
    full = ~np.isnan(E).any(0)
    w = np.linalg.lstsq(E[:, full].T, y_train.values[full], rcond=None)[0]
    print(f'lags 1-3 combined: accuracy {accuracy_score(pos[full], (w @ E[:, full]) > 0):.4f}')

    print('can the test dates be chained at any gap?')
    i_te, i_tr = np.where(is_test)[0], np.where(~is_test)[0]
    for g in args.gaps:
        Sg = overlap_similarity(V, available, gap=g)
        print(f'  gap {g:2d} days: test->any {Sg[i_te].max():.3f}, any->test {Sg[:, i_te].max():.3f}, '
              f'train->train {Sg[np.ix_(i_tr, i_tr)].max():.3f}')

    succ_tr, q_tr = match_successors(S[np.ix_(i_tr, i_tr)])
    seg, _ = chain_segments(succ_tr, q_tr)
    sizes = np.bincount(seg)
    print(f'training calendar: {len(sizes)} segments, largest {sorted(sizes.tolist(), reverse=True)[:6]}')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        bins = np.linspace(-0.2, 1.0, 61)
        ax[0].hist(best[~is_test], bins, alpha=0.75, label='best candidate')
        ax[0].hist(second[~is_test], bins, alpha=0.75, label='second best')
        ax[0].set(title='Training dates: similarity to candidate next day', xlabel='mean cosine over allocations',
                  ylabel='dates')
        ax[0].legend()
        ax[1].hist(quality[~is_test], bins, alpha=0.75, label='train dates')
        ax[1].hist(quality[is_test], bins, alpha=0.75, label='test dates')
        ax[1].axvline(0.85, color='k', ls='--', lw=1)
        ax[1].set_yscale('log')
        ax[1].set(title='Quality of the matched successor link', xlabel='similarity of the matched link',
                  ylabel='dates (log scale)')
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print('saved', args.plot)


if __name__ == '__main__':
    main()
