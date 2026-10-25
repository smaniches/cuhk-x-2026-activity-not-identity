"""Repair self-consistency violations by deselecting the conflicting multi option.

Evidence chain:
  1. Ground truth is self-consistent in 1238/1238 training groups (0 violations),
     so the constraint is a hard invariant, not a heuristic.
  2. The incumbent violates it in 9 of 195 test groups -> those are PROVABLE
     errors: at least one answer per group is wrong.
  3. Every violation has the same shape: multi/combination asserts action X is
     present, while single asserts X is absent (X is a non-chosen single option).
  4. Cross-question multiplicity (97.03% precision on train for `single`;
     incumbent already agrees with it 85/86 on test) breaks the tie: in 6 of 9
     it endorses the incumbent's SINGLE answer over X, in 3 it ties, and in
     ZERO does it favour X. So `single` is right and `multi` wrongly selected X.

Repair: deselect X from the multi answer in the 6 evidence-backed groups only.
Ties are left untouched. Nothing else in the file changes, so any leaderboard
delta is attributable to exactly these 6 rows.
"""
import csv
import collections

HERE = '/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution/'

# group -> (conflicting action X, multi qa_id)  [evidence-backed only, ties excluded]
# LM_test_0082 is deliberately EXCLUDED: there the present-assertion comes from
# `combination`, where the answer is a PAIR -- there is no single bit to
# deselect, so the repair is not the same unambiguous one-bit operation. Keeping
# it out leaves this experiment structurally uniform: five multi answers, each
# losing exactly one wrongly-selected option.
FIX = {
    'LM_test_0160': ('Stirring',       'test_0175'),
    'LM_test_0194': ('Using a phone',  'test_0208'),
    'LM_test_0199': ('Wiping surface', 'test_0213'),
    'LM_test_0208': ('Walking',        'test_0222'),
    'LM_test_0152': ('Using a phone',  'test_0605'),
}


def load(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        if '﻿qa_id' in r:
            r['qa_id'] = r.pop('﻿qa_id')
    return rows


te = load(HERE + 'test_qa.csv')
inc = {r['qa_id']: r['prediction']
       for r in csv.DictReader(open(HERE + 'incumbent_079239.csv'))}
by_qid = {r['qa_id']: r for r in te}

new = dict(inc)
changed = []
skipped = []
for grp, (X, mqid) in FIX.items():
    r = by_qid[mqid]
    assert r['category'] == 'multi', (mqid, r['category'])
    o = {L: r[L].strip() for L in 'ABCD'}
    letters = [L for L, v in o.items() if v == X]
    assert len(letters) == 1, (mqid, X, o)
    L = letters[0]
    cur = inc[mqid]
    assert L in cur, f"{mqid}: {L} not in incumbent answer {cur}"
    nxt = ''.join(sorted(c for c in cur if c != L))
    if not nxt:
        # multi answers are never empty in ground truth (checked on all 809
        # labelled train multi rows), so deselecting the ONLY selected option is
        # not a legal repair -- multi would have to pick a different option
        # instead, which is a second guess this experiment deliberately avoids.
        skipped.append((grp, mqid, X, cur))
        continue
    new[mqid] = nxt
    changed.append((grp, mqid, X, L, cur, nxt))

order = [r['qa_id'] for r in te]
out = HERE + 'cand_consistency_fix.csv'
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['qa_id', 'prediction'])
    for q in order:
        w.writerow([q, new[q]])

# ---- validation ----
chk = list(csv.DictReader(open(out)))
assert len(chk) == 682, len(chk)
assert [r['qa_id'] for r in chk] == order, "row order drift"
diff = [q for q in order if new[q] != inc[q]]
cat = {r['qa_id']: r['category'] for r in te}
assert set(diff) == {m for _, m, _, _, _, _ in changed}, "unexpected rows changed"
bad = [r for r in chk if not r['prediction']
       or any(c not in 'ABCD' for c in r['prediction'])
       or len(set(r['prediction'])) != len(r['prediction'])]
assert not bad, bad[:3]

for grp, mqid, X, L, cur, nxt in changed:
    print(f"  {grp:<14} {mqid}  multi {cur!r} -> {nxt!r}   (deselect {L}='{X}')")
for grp,mqid,X,cur in skipped:
    print(f"  SKIP {grp:<14} {mqid}  multi={cur!r} is single-option; cannot deselect '{X}'")
print(f"\nrows changed: {len(diff)}  categories: {collections.Counter(cat[q] for q in diff)}")
print("wrote", out)
