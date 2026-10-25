"""MECHANISM 4 — abstention recovery on `single` + `combination`.
                                            (LB: 0.87134 -> 0.87426, +2 items)

WHAT IT IS
action_decoder.py decides single/combination with a FIXED threshold, no sequence
anchor and no confidence gate, so it ABSTAINS on a large minority of rows.  This
module re-runs exactly those two categories with the two upgrades that the
`multi` arm proved out -- per-group threshold self-calibration and the
health == 1.0 gate -- and thereby DECIDES rows the crude rule left blank.

It is coverage, not correction.  Where the crude decoder and this one both fire
they agree on 100% of rows: 1198/1198 on TRAIN and 213/213 on the 2026 test file.

MEASURED ON TRAIN (groups re-derived unsupervised, labels used only to score;
size-2 groups simulated by dropping a member, mirroring the test file's pairs):

  | category            | crude fixed threshold      | self-calibrated + health==1 |
  | single              | fires 666/809, acc 0.9985  | fires 684/809, acc 1.0000   |
  | combination         | fires 532/790, acc 0.9962  | fires 665/790, acc 1.0000   |
  | single (pairs)      | fires 403/540, acc 0.9950  | fires 392/540, acc 1.0000   |
  | combination (pairs) | fires 301/527, acc 0.9867  | fires 379/527, acc 0.9947   |
  Pooled health == 1.0: 2096 rows, accuracy 0.9990.
  Rows firing inside UNHEALTHY groups: 360 rows, accuracy 0.9611 -- excluded.

INFERENCE-TIME INPUTS
  option texts A-D, `category`, `source`, clip ordering from `path`.  Nothing
  else.  No training file is opened at inference time.

CONSTANTS -- identical to multi_decoder.py, and for the same TRAIN-measured
reasons: alpha = 0.09 (tie-break only), health == 1.0, group size >= 2.
"""
from __future__ import annotations

from typing import Dict, Sequence

from common import group_action_counts, group_by_video, hits, segment_siblings
from multi_decoder import ALPHA, calibrate

TOUCH = ("single", "combination")


def decode(test_rows: Sequence[dict],
           alpha: float = ALPHA,
           min_group: int = 2,
           categories=TOUCH,
           source: str = "HAU",
           max_run: int = 3) -> Dict[str, str]:
    """Return {qa_id: letter} for every single/combination row the gate allows.

    Pure function of `test_rows`.  Emits a superset of action_decoder.decode on
    healthy groups; the caller decides precedence (they never conflict).
    """
    by_vid = group_by_video(r for r in test_rows if r.get("source") == source)
    groups = segment_siblings(test_rows, source=source, max_run=max_run)

    out: Dict[str, str] = {}
    for g in groups:
        if len(g) < min_group:
            continue
        rows_in = [r for v in g for r in by_vid[v]]
        counts, slots, rows, anchor = group_action_counts(rows_in)
        _thr, present, health = calibrate(counts, slots, rows, anchor,
                                          alpha=alpha, touch=categories)
        if health < 0.999:
            continue
        for row in rows:
            if row.get("category") not in categories:
                continue
            h = hits(row, present)
            if len(h) == 1:
                out[row["qa_id"]] = h[0]
    return out


__all__ = ["decode"]
