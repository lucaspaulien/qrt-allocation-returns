"""Recovering the true order of the anonymised dates.

The date labels DATE_xxxx are shuffled. But if t' is the day after t, then for
every allocation the 20-day return window slides by one:

    RET_{i+1}(t') = RET_i(t),   i = 1..19

so the last 19 returns of t' are the first 19 of t. The organisers added a
little noise to every value, which rules out exact matching, but averaging a
cosine similarity over the ~200 allocations present on both dates makes the
true successor stand out by a huge margin (best 0.98 vs second best 0.20 on
the real data). Solving the resulting one-successor-per-date assignment
problem rebuilds the calendar.

The point of this is *validation*, not prediction: with the calendar known,
cross-validation folds can be made of consecutive days with an embargo, which
is impossible with shuffled labels. (It would also give the target away -- the
next day's RET_1 *is* today's target -- but the test dates turn out to have no
neighbour within 12 days, so nothing leaks into the test set.)
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .data import RET


def return_tensor(frames):
    """Stack one or more row frames into a (date, allocation, lag) tensor.

    Returns (dates, allocations, V, available) where V[d, a, k] = RET_{k+1} and
    ``available[d, a]`` is True when the pair exists with a full return window.
    """
    A = pd.concat(frames)
    dates = np.sort(A.TS.unique())
    allocs = np.sort(A.ALLOCATION.unique())
    d_pos = {d: i for i, d in enumerate(dates)}
    a_pos = {a: i for i, a in enumerate(allocs)}
    V = np.full((len(dates), len(allocs), len(RET)), np.nan)
    V[A.TS.map(d_pos).values, A.ALLOCATION.map(a_pos).values] = A[RET].to_numpy(float)
    return dates, allocs, V, ~np.isnan(V).any(2)


def _unit_block(V, available, lags):
    """Each (date, allocation) window slice scaled to unit norm, flattened per date."""
    M = np.nan_to_num(V[:, :, lags])
    M = M / np.maximum(np.linalg.norm(M, axis=2, keepdims=True), 1e-300)
    M[~available] = 0
    return M.reshape(V.shape[0], -1)


def overlap_similarity(V, available, gap=1):
    """S[i, j] = mean cosine over shared allocations between "date j is ``gap`` days after date i".

    With a gap of g days the windows overlap on 20 - g lags: RET_1..RET_{20-g}
    of date i against RET_{1+g}..RET_20 of date j. The diagonal is set to -9.
    """
    n_lags = V.shape[2]
    shared = available.astype(float) @ available.astype(float).T
    S = (_unit_block(V, available, slice(0, n_lags - gap))
         @ _unit_block(V, available, slice(gap, n_lags)).T) / np.maximum(shared, 1)
    np.fill_diagonal(S, -9)
    return S


def match_successors(S):
    """One successor per date maximising total similarity (Hungarian algorithm).

    Returns (succ, quality): ``succ[i]`` is the matched next date of i and
    ``quality[i]`` the similarity of that link. The last day of each true run
    gets *some* successor too, with a low quality, which is how breaks are found.
    """
    rows, cols = linear_sum_assignment(-S)
    succ = np.empty(len(S), int)
    succ[rows] = cols
    return succ, S[np.arange(len(S)), succ]


def chain_segments(succ, quality, threshold=0.85):
    """Cut the successor chain at weak links into runs of consecutive days.

    Returns (segment, rank): the segment id of each date and its position
    inside the segment (0 = first day). Within a segment, rank differences are
    true distances in trading days; across segments distances are unknown.
    """
    n = len(succ)
    nxt = np.where(quality > threshold, succ, -1)
    prev = np.full(n, -1)
    for i, j in enumerate(nxt):
        if j >= 0:
            prev[j] = i
    segment = np.full(n, -1)
    rank = np.full(n, -1)
    s = 0
    # start from dates without a predecessor; any leftover is a closed cycle
    for start in list(np.where(prev == -1)[0]) + list(range(n)):
        if segment[start] != -1:
            continue
        j, p = start, 0
        while j != -1 and segment[j] == -1:
            segment[j], rank[j] = s, p
            p += 1
            j = nxt[j]
        s += 1
    return segment, rank


def reconstruct(X_train, threshold=0.85):
    """Chronology of the training dates: DataFrame with columns TS, segment, rank."""
    dates, _, V, available = return_tensor([X_train])
    succ, quality = match_successors(overlap_similarity(V, available))
    segment, rank = chain_segments(succ, quality, threshold)
    return pd.DataFrame({'TS': dates, 'segment': segment, 'rank': rank})


def target_from_successors(V, succ, d_idx, a_idx, max_lag=3):
    """Read each row's target back out of the next dates' return windows.

    The target of date t reappears as RET_1 of t+1, RET_2 of t+2, ... Returns
    an array (max_lag, n_rows) of these noisy copies (NaN where missing).
    This is the leak the reconstruction would expose -- used here only to
    measure how well the chain is recovered.
    """
    step = np.arange(len(succ))
    out = []
    for k in range(max_lag):
        step = succ[step]
        out.append(V[step, :, k][d_idx, a_idx])
    return np.array(out)
