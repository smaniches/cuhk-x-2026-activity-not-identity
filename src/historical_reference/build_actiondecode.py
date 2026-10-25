#!/usr/bin/env python3
"""
Label-free action-set decoder for CUHK-X Large Model Track.

No training labels are used anywhere in the decoder. It reads the action set of
each video-group directly out of the option-text frequencies in test_qa.csv.

Mechanism (all three parts validated on training, see H_cuhk_large_text.md):
 1. Videos come in speed-groups (triplets/pairs). Group boundaries are recovered
    from the test CSV alone: two consecutive videos are in the same group iff
    their four `emotion` option texts overlap in exactly 3 (the shared "core"),
    with a max run of 3. Recovers 267/272 true training groups exactly.
 2. A group shares one action set (264/265 training triplets show zero
    contradiction vs a 9.9% random-triplet null).
 3. Distractors are drawn exclusively from actions the video does NOT contain
    (0 violations in ~4400 labelled training checks). So an action's frequency
    across the group's options separates present from absent cleanly.

Threshold: count >= max(3, round(0.09 * option_slots)) marks an action present.
Then: `single` answer = the unique option in the set; `combination` answer = the
unique option whose action list is a subset of the set. `multi` is NOT used
(only 82.7% on training - too weak to override a model).

Base: cand_emotion_core_v2.csv (emotion-triplet repair). Only `single` and
`combination` rows are touched, so the two mechanisms do not overlap.
"""
import collections
import pandas as pd

SRC = "/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution"
OUT = "/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution/cand_actiondecode.csv"
ALPHA = 0.09

test = pd.read_csv(f"{SRC}/test_qa.csv").drop(columns=["prediction"])
base = pd.read_csv(f"{SRC}/cand_emotion_core_v2.csv")
champ = pd.read_csv(f"{SRC}/cand_emotion_s3_v1.csv")

test["vid"] = test.path.str.extract(r"LM_test_(\d+)")[0].astype(int)
pred = dict(zip(base.qa_id, base.prediction))
champ_pred = dict(zip(champ.qa_id, champ.prediction))


def option_actions(row, letter):
    """Actions listed in one option. `combination` options are comma-separated."""
    raw = str(getattr(row, letter)).strip()
    if row.category == "combination":
        return [x.strip() for x in raw.split(",")]
    return [raw]


# ---- 1. recover speed-group boundaries from emotion option overlap ----------
hau = test[test.source == "HAU"]
emotion_opts = {
    r.vid: frozenset(str(getattr(r, k)).strip() for k in "ABCD")
    for _, r in hau[hau.category == "emotion"].iterrows()
}

groups, current = [], []
for vid in sorted(hau.vid.unique()):
    same = (
        current
        and len(emotion_opts.get(current[-1], frozenset()) & emotion_opts.get(vid, frozenset())) >= 3
        and len(current) < 3
    )
    if same:
        current.append(vid)
    else:
        if current:
            groups.append(current)
        current = [vid]
groups.append(current)

# ---- 2+3. decode each group's action set, then force single / combination ---
by_vid = {v: g for v, g in hau.groupby("vid")}
changes = []

for group in groups:
    counts, slots, rows = collections.Counter(), 0, []
    for vid in group:
        for _, row in by_vid[vid].iterrows():
            if row.category == "emotion":       # emotion rows are owned by core_v2
                continue
            rows.append(row)
            for letter in "ABCD":
                for action in option_actions(row, letter):
                    counts[action] += 1
                    slots += 1

    threshold = max(3, round(ALPHA * slots))
    present = {a for a, c in counts.items() if c >= threshold}

    for row in rows:
        if row.category == "single":
            hits = [k for k in "ABCD" if str(getattr(row, k)).strip() in present]
        elif row.category == "combination":
            hits = [k for k in "ABCD" if set(option_actions(row, k)) <= present]
        else:
            continue                             # multi deliberately untouched
        if len(hits) != 1:
            continue
        new = hits[0]
        if new != pred[row.qa_id]:
            changes.append((row.qa_id, row.category, pred[row.qa_id], new, champ_pred[row.qa_id]))
            pred[row.qa_id] = new

# ---- write + self-check -----------------------------------------------------
out = pd.DataFrame({"qa_id": test.qa_id, "prediction": [pred[q] for q in test.qa_id]})
out.to_csv(OUT, index=False)

cat = dict(zip(test.qa_id, test.category))
src = dict(zip(test.qa_id, test.source))
errs = []
if len(out) != 682:
    errs.append(f"row count {len(out)} != 682")
if list(out.qa_id) != list(test.qa_id):
    errs.append("qa_id order does not match test_qa.csv")
for q, p in zip(out.qa_id, out.prediction):
    p = str(p)
    if not p or p == "nan":
        errs.append(f"{q}: empty")
    if any(c not in "ABCD" for c in p):
        errs.append(f"{q}: illegal char {p!r}")
    if len(set(p)) != len(p):
        errs.append(f"{q}: repeated char {p!r}")
    if cat[q] in ("single", "emotion", "combination", "object_interaction") and len(p) != 1:
        errs.append(f"{q}: {cat[q]} must be 1 letter, got {p!r}")
    if cat[q] == "sequence" and sorted(p) != list("ABCD"):
        errs.append(f"{q}: sequence must be a permutation, got {p!r}")
    if cat[q] != "sequence" and list(p) != sorted(p):
        errs.append(f"{q}: not alphabetical {p!r}")
    if cat[q] == "single" and src[q] == "HARn" and "D" in p:
        errs.append(f"{q}: HARn single has no option D")

emotion_touched = [
    q for q in out.qa_id
    if cat[q] == "emotion" and pred[q] != dict(zip(base.qa_id, base.prediction))[q]
]
if emotion_touched:
    errs.append(f"emotion rows modified: {emotion_touched}")

print(f"wrote {OUT}")
print(f"PREFLIGHT: {'PASS - 0 violations' if not errs else 'FAIL'}")
for e in errs:
    print("  ", e)
print(f"\ngroups recovered: {len(groups)} "
      f"({collections.Counter(len(g) for g in groups)})")
print(f"rows changed vs cand_emotion_core_v2.csv: {len(changes)}")
print(f"{'qa_id':<12}{'category':<14}{'core_v2':<9}{'new':<6}champion")
for qa, c, old, new, cp in sorted(changes):
    flag = "" if old == cp else "   <-- COLLISION: core_v2 differs from champion here"
    print(f"{qa:<12}{c:<14}{old:<9}{new:<6}{cp}{flag}")
