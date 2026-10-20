"""Refit on the whole training set and write the 0/1 submission file.

    python scripts/make_submission.py --data-dir data --out submission.csv
"""
import argparse

import pandas as pd

from allocpred.data import default_data_dir, load
from allocpred.pipeline import Data, fit_predict_test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=str(default_data_dir()))
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()

    X_train, y_train, X_test, sample = load(args.data_dir)
    score = fit_predict_test(Data(X_train, y_train, X_test), num_threads=args.threads)
    sub = pd.DataFrame({'ROW_ID': X_test.index, 'prediction': (score > 0).astype(int)})
    assert list(sub.ROW_ID) == list(sample.ROW_ID)
    sub.to_csv(args.out, index=False)
    print(f'wrote {args.out}: {len(sub)} rows, {sub.prediction.mean():.4f} predicted positive')


if __name__ == '__main__':
    main()
