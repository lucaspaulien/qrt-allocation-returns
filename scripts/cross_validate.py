"""Purged chronological 5-fold CV of the ridge, the three LightGBM heads and the blend.

    python scripts/cross_validate.py --importance-plot docs/feature_importance.png

The full run (3 heads x 3 seeds x 5 folds) takes ~3-4 min on a recent laptop;
--quick uses one seed per head.
"""
import argparse
import time

import numpy as np
import pandas as pd

from allocpred.chronology import reconstruct
from allocpred.data import default_data_dir, load
from allocpred.pipeline import Data, accuracy, blend, cross_validate
from allocpred.validation import chronological_folds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=str(default_data_dir()))
    ap.add_argument('--out', default='docs/cv_results.csv')
    ap.add_argument('--importance-plot', default=None)
    ap.add_argument('--embargo', type=int, default=21)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--quick', action='store_true', help='one seed per LightGBM head')
    args = ap.parse_args()

    t0 = time.time()
    X_train, y_train, X_test, _ = load(args.data_dir)
    chrono = reconstruct(X_train)
    fold, seg, rank = chronological_folds(chrono, X_train.TS.values)
    print(f'{chrono.segment.nunique()} calendar segments; fold sizes {np.bincount(fold).tolist()}', flush=True)
    data = Data(X_train, y_train, X_test)
    seeds = (1,) if args.quick else (1, 2, 3)
    oof, imp = cross_validate(data, fold, seg, rank, embargo=args.embargo, seeds=seeds,
                              num_threads=args.threads, log=lambda s: print(s, flush=True))

    acc = accuracy(oof, data.sign)
    rows = [dict(model=m, accuracy=a) for m, a in acc.items()]
    b = blend(oof)
    per_fold = [np.mean((b[fold == k] > 0) == data.sign[fold == k].astype(bool)) for k in np.unique(fold)]
    rows += [dict(model=f'blend_fold{k}', accuracy=a) for k, a in enumerate(per_fold)]
    rows += [dict(model='always_positive', accuracy=data.sign.mean()),
             dict(model='sign_of_RET_1', accuracy=np.mean((data.ret1 > 0) == data.sign.astype(bool)))]
    for q in [0.40, 0.45, 0.50, 0.55, 0.60]:
        rows.append(dict(model=f'blend_threshold_q{q:.2f}',
                         accuracy=np.mean((b > np.quantile(b, q)) == data.sign.astype(bool))))
    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    print(res.round(4).to_string(index=False))
    print(f'blend fold std {np.std(per_fold):.4f}; total {(time.time() - t0) / 60:.1f} min')

    if args.importance_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        gain = pd.DataFrame(imp['mse'], columns=imp['columns'])
        gain = gain / gain.values.sum(1, keepdims=True)
        top = gain.mean().sort_values(ascending=False).index[:20][::-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(top, gain[top].mean(), xerr=gain[top].std(), color='#4C72B0')
        ax.set(xlabel='share of total gain (mse head, fold 0, mean over seeds)',
               title='Top 20 features')
        fig.tight_layout()
        fig.savefig(args.importance_plot, dpi=130)
        print('saved', args.importance_plot)


if __name__ == '__main__':
    main()
