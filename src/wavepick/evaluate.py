"""Scoring detected peaks against ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Match:
    tp: int
    fp: int
    fn: int
    pairs: list[tuple[int, int]]  #: (detected index, truth index)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_peaks(detected: np.ndarray, truth: np.ndarray, tol: float) -> Match:
    """Greedy one-to-one matching of positions, closest pairs first.

    Each true peak is matched to at most one detected peak within ``tol`` and
    vice versa. Extra detections of the same true peak count as false positives.
    """
    d = np.asarray(detected, dtype=float)
    t = np.asarray(truth, dtype=float)
    if d.size == 0 or t.size == 0:
        return Match(tp=0, fp=d.size, fn=t.size, pairs=[])
    dist = np.abs(d[:, None] - t[None, :])
    cand = np.argwhere(dist <= tol)
    order = np.argsort(dist[cand[:, 0], cand[:, 1]], kind="stable")
    used_d, used_t, pairs = set(), set(), []
    for i, j in cand[order]:
        if i not in used_d and j not in used_t:
            used_d.add(int(i))
            used_t.add(int(j))
            pairs.append((int(i), int(j)))
    tp = len(pairs)
    return Match(tp=tp, fp=d.size - tp, fn=t.size - tp, pairs=pairs)
