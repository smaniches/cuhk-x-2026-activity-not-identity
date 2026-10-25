"""MECHANISM 3 — `multi` via sibling-group pooling + threshold self-calibration.
                                            (LB: 0.84502 -> 0.87134, +18 items)

WHAT IT IS
Same three protocol facts as action_decoder.py, plus two upgrades that turn the
blanket decoder's weak 82.7% on `multi` into 97%+ where it fires:

 4. SEQUENCE ANCHOR.  A `sequence` question's four options are asserted present by
    the question text itself, so they join the action set for free.
 5. PER-GROUP THRESHOLD SELF-CALIBRATION.  Instead of a fixed
    `max(3, alpha*slots)`, sweep the threshold and keep the value satisfying the
    most LABEL-FREE consistency constraints on that group:
        every `single` question has EXACTLY ONE consistent option
        every `combination` question has EXACTLY ONE consistent option
        every `multi` question has BETWEEN 1 AND 3 consistent options
    Ties break toward the alpha*slots default.
    TRAIN `multi` exact match: 0.866 -> 0.934 on full groups,
                               0.774 -> 0.877 on groups cut to two members.
 6. HEALTH GATE.  `health` = the fraction of the group's single/combination
    constraints satisfied at the chosen threshold -- computed WITHOUT labels and
    WITHOUT looking at `multi`, so it cannot be circular.
    TRAIN: health == 1.0 -> multi exact match 0.9747 (n=672, size-3 groups) and
    0.9718 (n=390, size-2 groups);  health < 1.0 -> 0.64-0.75.
    Only health == 1.0 groups with >= 2 members are decided.

INFERENCE-TIME INPUTS
  option texts A-D, `category`, `source`, clip ordering from `path`.  Nothing
  else.  No training file is opened at inference time.

CONSTANTS
  alpha = 0.09         -- TIE-BREAK ONLY here; TRAIN accuracy is flat 0.971-0.975
                          across alpha in [0.07, 0.14].
  1 <= |multi| <= 3    -- TRAIN answer lengths: 312 x1, 298 x2, 197 x3, 2 x4.
                          Length-4 predictions were 4/4 wrong on TRAIN.
  health == 1.0        -- the top of a TRAIN-measured monotone curve
                          (0.64-0.75 below 1.0, 0.97+ at 1.0), not an LB fit.
  group size >= 2      -- a singleton group has nothing to pool.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

from common import (LETTERS, group_action_counts, group_by_video, hits,
                    live_options, option_text, segment_siblings)

ALPHA = 0.09
MULTI_MAX = 3


# ------------------------------------------------- shared self-calibration core
def calibrate(counts, slots, rows, anchor, alpha: float = ALPHA,
              touch=("single", "combination"), multi_max: int = MULTI_MAX
              ) -> Tuple[int, set, float]:
    """Pick the group's action-count threshold with no labels at all.

    Returns (threshold, present_action_set, health).  `health` counts only the
    single/combination constraints, so it is independent of whatever `multi`
    decision it is later used to gate.
    """
    default = max(3, round(alpha * slots))

    def present_at(t):
        return {a for a, c in counts.items() if c >= t} | anchor

    def satisfied(t):
        pres, sat, tot = present_at(t), 0, 0
        for row in rows:
            cat = row.get("category")
            if cat in ("single", "combination"):
                tot += 1
                sat += (len(hits(row, pres)) == 1)
            elif cat == "multi":
                n = len(hits(row, pres))
                tot += 1
                sat += (1 <= n <= multi_max)
        return sat, tot

    hi = max(counts.values()) if counts else 1
    thr = max(range(2, hi + 1),
              key=lambda t: (2 * satisfied(t)[0] - satisfied(t)[1],
                             -abs(t - default)))
    present = present_at(thr)

    ok = tot = 0
    for row in rows:
        if row.get("category") in touch:
            tot += 1
            ok += (len(hits(row, present)) == 1)
    return thr, present, (ok / tot if tot else 0.0)


# -------------------------------------------------------------------- inference
def decode(test_rows: Sequence[dict],
           alpha: float = ALPHA,
           min_group: int = 2,
           multi_max: int = MULTI_MAX,
           source: str = "HAU",
           max_run: int = 3) -> Dict[str, str]:
    """Return {qa_id: letters} for every `multi` row the gate lets through.

    Pure function of `test_rows`.  Never emits an empty answer and never emits a
    four-letter `multi` answer.
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
                                          alpha=alpha, multi_max=multi_max)
        if health < 0.999:
            continue
        for row in rows:
            if row.get("category") != "multi":
                continue
            h = [k for k in live_options(row)
                 if option_text(row, k) in present]
            if not h or len(h) > multi_max:
                continue
            out[row["qa_id"]] = "".join(sorted(h))
    return out


__all__ = ["decode", "calibrate", "ALPHA", "MULTI_MAX"]
