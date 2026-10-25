"""MECHANISM 1 — emotion triplet assignment.   (LB: 0.80994 -> 0.83625, +18 items)

WHAT IT IS
Each scripted session is filmed three times at three different speeds (the train
path suffix `HAU/userN/a-b-c` has c in {1,2,3} = the trial index).  The annotator
is handed ONE pool of three manner adverbs for the session -- the "core" -- and
assigns one to each trial, slow trial first.  Ground truth therefore satisfies two
constraints that no per-row model respects:

    |core| == 3                                  262/265 train sessions
    the three answers in a session are DISTINCT  265/265 train sessions   (HARD)
    every answer lies in the core                262/262 train sessions   (HARD)

So the emotion question is a 3x3 ASSIGNMENT problem, not three independent picks.

INFERENCE-TIME INPUTS
  * the four `emotion` option texts of every clip           (test file)
  * clip ordering within `source == "HAU"`                  (path)
  * category / source columns                               (test file)
  * ONE parameter table fitted on TRAIN: P(trial slot | adverb).

TRAINED PARAMETER
`fit_slot_table` counts, over the training `emotion` rows, how often each adverb
is the answer at each trial index.  This is a property of the annotation protocol
(the adverb vocabulary's speed ordering), not of any subject, which is why it
survives the subject-disjoint split.  Nothing else is fitted.

CONSTANTS AND WHERE THEY COME FROM
  link Jaccard >= 0.40  -- TRAIN measurement: adjacent-pair option-set Jaccard is
                           0.603 within a session vs 0.062 across sessions.  Any
                           value in (0.10, 0.55) gives the same segmentation.
  max run  = 3          -- recording protocol (three trials per session).
  |core|  == 3          -- TRAIN, 262/265.
  add-one smoothing     -- standard Laplace prior, not tuned.

DELIBERATELY NOT APPLIED to groups of size < 3: with one trial missing the slot
assignment is unidentifiable (measured: marginalising over the three slot
hypotheses scores 0.557 per row, i.e. worse than doing nothing).
"""
from __future__ import annotations

import itertools
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from common import (LETTERS, group_by_video, live_options, option_set,
                    option_text, segment_siblings, video_key)

TRIAL_SUFFIX = re.compile(r"-(\d+)\s*$")


# --------------------------------------------------------------------- training
def fit_slot_table(train_rows: Sequence[dict],
                   category: str = "emotion",
                   source: str = "HAU") -> Dict[str, Counter]:
    """P(trial slot | adverb), counted on TRAIN only.

    The trial index is read off the training path suffix (`.../a-b-c` -> c).
    Returns {adverb: Counter({slot: count})}.
    """
    table: Dict[str, Counter] = defaultdict(Counter)
    for r in train_rows:
        if r.get("category") != category or r.get("source") != source:
            continue
        m = TRIAL_SUFFIX.search(str(r["path"]))
        ans = str(r.get("answer", "")).strip()
        if not m or ans not in LETTERS:
            continue
        table[option_text(r, ans)][int(m.group(1))] += 1
    return dict(table)


def n_slots(table: Dict[str, Counter], default: int = 3) -> int:
    seen = {s for c in table.values() for s in c}
    return max(seen) if seen else default


# -------------------------------------------------------------------- inference
def _logp(table, word, slot, n, alpha=1.0):
    c = table.get(word)
    num = (c[slot] if c else 0) + alpha
    den = (sum(c.values()) if c else 0) + alpha * n
    return math.log(num / den)


def decode(test_rows: Sequence[dict],
           slot_table: Dict[str, Counter],
           source: str = "HAU",
           category: str = "emotion",
           link_jaccard: float = 0.40,
           max_run: int = 3) -> Dict[str, str]:
    """Return {qa_id: letter} for every emotion row the mechanism can decide.

    Pure function of (test_rows, slot_table).  Touches no other category, decides
    nothing outside complete groups, and enumerates no ids.
    """
    n = n_slots(slot_table, max_run)
    rows = [r for r in test_rows
            if r.get("source") == source and r.get("category") == category]
    by_vid = group_by_video(rows)
    opts = {v: option_set(rs[0]) for v, rs in by_vid.items()}

    groups = _segment(sorted(by_vid, key=_natural), opts, link_jaccard, max_run)

    out: Dict[str, str] = {}
    for g in groups:
        if len(g) != max_run:                  # incomplete session -> abstain
            continue
        core = set.intersection(*(set(opts[v]) for v in g))
        if len(core) != max_run:               # core not recoverable -> abstain
            continue
        words = sorted(core)
        best = max(itertools.permutations(words),
                   key=lambda p: sum(_logp(slot_table, p[i], i + 1, n)
                                     for i in range(len(p))))
        for i, v in enumerate(g):
            row = by_vid[v][0]
            for k in live_options(row):
                if option_text(row, k) == best[i]:
                    out[row["qa_id"]] = k
                    break
    return out


def _natural(key: str):
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key))


def _segment(order, opts, jt, max_run) -> List[List[str]]:
    """Link consecutive clips whose emotion option sets are Jaccard-close."""
    def jac(a, b):
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    groups, cur = [], []
    for v in order:
        if cur and jac(opts[cur[-1]], opts[v]) >= jt and len(cur) < max_run:
            cur.append(v)
        else:
            if cur:
                groups.append(cur)
            cur = [v]
    if cur:
        groups.append(cur)
    return groups




# ------------------------------------------------------- duration (pace) signal
# MECHANISM 5 — emotion pace from clip duration.  (LB: 0.87426 -> 0.90935, +12 rows)
# Emotion adverbs describe PACE, and a triple is one script filmed at three speeds,
# so speed is a quantity we can MEASURE rather than infer combinatorially.  Clip
# duration is the frame count, readable from ZIP metadata or from an mp4 `stsz`
# atom -- no decoding, and available even on a Depth-only DATA_DIR.
# LOO-by-user over 260 training triples:
#     positional slot alone  0.8244   (the mechanism above)
#     duration rank alone    0.8423
#     positional + duration  0.9346   <- both terms, this function
# They disagree on 36.5% of triples because the trial index is a FILMING SLOT
# while duration is a MEASURED SPEED: genuinely different information.

def _suffixes(key: str):
    parts = [x for x in str(key).replace("\\", "/").split("/") if x]
    for n in (3, 2):
        if len(parts) >= n:
            yield "/".join(parts[-n:])


def build_duration_index(durations: Dict[str, float]) -> Dict[str, float]:
    """Index durations by path SUFFIX so lookup survives a different DATA_DIR root.

    The live organizer tree may root paths differently from ours (ours arrive as
    `Training/HAU/user9/7-2-1` while the QA `path` column says `HAU/user9/7-2-1`).
    Longer suffixes are preferred, so HAU and HARn cannot collide on `userN/a-b-c`.
    """
    idx: Dict[str, float] = {}
    for k, v in durations.items():
        for sfx in _suffixes(k):
            idx.setdefault(sfx, v)
    return idx


def lookup_duration(index: Dict[str, float], path: str):
    for sfx in _suffixes(path):
        if sfx in index:
            return index[sfx]
    return None


def fit_duration_table(train_rows: Sequence[dict],
                       durations: Dict[str, float],
                       category: str = "emotion",
                       source: str = "HAU",
                       max_run: int = 3) -> Dict[str, Counter]:
    """P(duration rank within its session | adverb), counted on TRAIN only."""
    _idx = build_duration_index(durations)
    sessions: Dict[str, list] = defaultdict(list)
    for r in train_rows:
        if r.get("category") != category or r.get("source") != source:
            continue
        m = TRIAL_SUFFIX.search(str(r["path"]))
        ans = str(r.get("answer", "")).strip()
        if not m or ans not in LETTERS:
            continue
        d = (lookup_duration(_idx, video_key(str(r["path"])))
             or lookup_duration(_idx, str(r["path"])))
        if d is None:
            continue
        sessions[str(r["path"])[: str(r["path"]).rfind("-")]].append(
            (d, option_text(r, ans)))
    table: Dict[str, Counter] = defaultdict(Counter)
    for members in sessions.values():
        if len(members) != max_run:
            continue
        for rank, (_, word) in enumerate(sorted(members), start=1):
            table[word][rank] += 1
    return dict(table)


def decode_with_duration(test_rows: Sequence[dict],
                         slot_table: Dict[str, Counter],
                         durations: Dict[str, float],
                         dur_table: Dict[str, Counter],
                         source: str = "HAU",
                         category: str = "emotion",
                         link_jaccard: float = 0.40,
                         max_run: int = 3) -> Dict[str, str]:
    """As `decode`, but scores each permutation with BOTH terms.

    Falls back to positional-only for any group whose durations are unavailable,
    so a DATA_DIR without readable clips degrades instead of failing.
    """
    n = n_slots(slot_table, max_run)
    nd = n_slots(dur_table, max_run) if dur_table else max_run
    _didx = build_duration_index(durations)
    rows = [r for r in test_rows
            if r.get("source") == source and r.get("category") == category]
    by_vid = group_by_video(rows)
    opts = {v: option_set(rs[0]) for v, rs in by_vid.items()}
    groups = _segment(sorted(by_vid, key=_natural), opts, link_jaccard, max_run)

    out: Dict[str, str] = {}
    for g in groups:
        if len(g) != max_run:
            continue
        core = set.intersection(*(set(opts[v]) for v in g))
        if len(core) != max_run:
            continue
        words = sorted(core)
        dvals = {v: (lookup_duration(_didx, video_key(str(by_vid[v][0]["path"])))
                     or lookup_duration(_didx, str(by_vid[v][0]["path"]))) for v in g}
        rank_of = {}
        if all(dvals[v] is not None for v in g):
            for rk, v in enumerate(sorted(g, key=lambda x: dvals[x]), start=1):
                rank_of[v] = rk

        def score(p):
            tot = sum(_logp(slot_table, p[i], i + 1, n) for i in range(len(p)))
            if rank_of:
                tot += sum(_logp(dur_table, p[i], rank_of[g[i]], nd)
                           for i in range(len(p)))
            return tot

        best = max(itertools.permutations(words), key=score)
        for i, v in enumerate(g):
            row = by_vid[v][0]
            for k in live_options(row):
                if option_text(row, k) == best[i]:
                    out[row["qa_id"]] = k
                    break
    return out


__all__ = ["fit_slot_table", "decode", "fit_duration_table",
           "decode_with_duration", "build_duration_index", "lookup_duration"]
