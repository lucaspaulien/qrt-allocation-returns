"""Loading the challenge files.

The raw data (~300 MB) is not redistributed with this repository. Download it
from the challenge page and put the four files in ``data/`` under their
standard names: X_train.csv, y_train.csv, X_test.csv, sample_submission.csv.

Each row is one (date, allocation) pair. ``RET_1 .. RET_20`` are the
allocation's returns over the previous 20 days (``RET_1`` = yesterday),
``SIGNED_VOLUME_*`` the matching signed traded volumes. The target is the next
day's return; the challenge scores the accuracy of its predicted sign.
"""
import os
from pathlib import Path

import pandas as pd

RET = [f'RET_{i}' for i in range(1, 21)]
SIGNED_VOLUME = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]


def default_data_dir():
    return Path(os.environ.get('ALLOCPRED_DATA', 'data'))


def load(data_dir):
    """Return X_train, y_train (Series), X_test, sample_submission."""
    d = Path(data_dir)
    X_train = pd.read_csv(d / 'X_train.csv', index_col='ROW_ID')
    X_test = pd.read_csv(d / 'X_test.csv', index_col='ROW_ID')
    y_train = pd.read_csv(d / 'y_train.csv', index_col='ROW_ID').target
    sample = pd.read_csv(d / 'sample_submission.csv')
    return X_train, y_train, X_test, sample
