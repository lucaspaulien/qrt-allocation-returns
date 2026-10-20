import numpy as np

from allocpred.chronology import (chain_segments, match_successors, overlap_similarity, reconstruct,
                                  return_tensor, target_from_successors)


def true_next(X, dates):
    """Date label of the true next day for each entry of ``dates`` (None for the last)."""
    day = X.groupby('TS').true_day.first()
    by_day = {d: ts for ts, d in day.items()}
    return [by_day.get(day[ts] + 1) for ts in dates]


def test_labels_are_shuffled(panel):
    X, _ = panel
    day = X.groupby('TS').true_day.first().sort_index()
    assert abs(np.corrcoef(np.arange(len(day)), day.values)[0, 1]) < 0.3


def test_successor_recovered_despite_noise(panel):
    X, _ = panel
    dates, _, V, avail = return_tensor([X])
    succ, quality = match_successors(overlap_similarity(V, avail))
    expected = true_next(X, dates)
    has_next = np.array([e is not None for e in expected])
    found = np.array([dates[s] == e for s, e in zip(succ, expected)])
    assert found[has_next].all()
    assert (quality[has_next] > 0.95).all()
    assert quality[~has_next].max() < 0.5          # the last day has no true successor


def test_similarity_at_larger_gaps(panel):
    X, _ = panel
    dates, _, V, avail = return_tensor([X])
    day = X.groupby('TS').true_day.first().reindex(dates).values
    S = overlap_similarity(V, avail, gap=5)
    i = np.where(day == 10)[0][0]
    assert day[S[i].argmax()] == 15


def test_segments_follow_calendar_breaks(broken_panel):
    X, _ = broken_panel
    chrono = reconstruct(X)
    day = X.groupby('TS').true_day.first()
    chrono['day'] = chrono.TS.map(day)
    assert chrono.segment.nunique() == 3
    for _, g in chrono.groupby('segment'):
        g = g.sort_values('rank')
        # inside a segment rank differences are true distances in days
        assert (np.diff(g.day.values) == 1).all()
        assert (np.diff(g['rank'].values) == 1).all()


def test_chain_segments_on_hand_built_chain():
    succ = np.array([1, 2, 0, 4, 3])
    q = np.array([0.99, 0.99, 0.10, 0.99, 0.20])     # 0 -> 1 -> 2 | 3 -> 4
    seg, rank = chain_segments(succ, q)
    assert seg[0] == seg[1] == seg[2] != seg[3] == seg[4]
    assert rank.tolist() == [0, 1, 2, 0, 1]


def test_target_reappears_in_successor_windows(panel):
    X, y = panel
    dates, allocs, V, avail = return_tensor([X])
    succ, _ = match_successors(overlap_similarity(V, avail))
    d_idx = X.TS.map({d: i for i, d in enumerate(dates)}).values
    a_idx = X.ALLOCATION.map({a: i for i, a in enumerate(allocs)}).values
    E = target_from_successors(V, succ, d_idx, a_idx, max_lag=2)
    ok = ~np.isnan(E[0])
    assert np.mean(np.abs(E[0][ok] - y.values[ok]) < 1e-3) > 0.97
