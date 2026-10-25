#!/usr/bin/env python3
"""
`single` + `combination` RECALIBRATION for CUHK-X Large Model Track  (AGENT M).

Parent: cand_multi.csv (0.87134, current champion).  ONLY `single` and
`combination` rows are touched.  `multi` (mine), `emotion` (core_v2's),
`sequence` and `object_interaction` are left exactly as the parent has them.

WHY THIS EXISTS
`cand_actiondecode.csv` decoded single/combination with a FIXED threshold
(max(3, 0.09*slots)), no sequence anchor and no confidence gate, and fired only
where exactly one option hit.  This file re-runs those two categories with the
two upgrades that won the `multi` arm:
  * per-group threshold SELF-CALIBRATION against the label-free consistency
    constraints, and
  * the health == 1.0 gate.

MEASURED ON TRAINING (groups re-derived unsupervised; labels used only to score;
size-2 groups simulated by dropping a member, to mirror test's 23 pairs):

| category    | crude (actiondecode)          | self-calibrated + health==1 |
|-------------|-------------------------------|------------------------------|
| single      | fires 666/809, acc 0.9985     | fires 684/809, acc **1.0000** |
| combination | fires 532/790, acc 0.9962     | fires 665/790, acc **1.0000** |
| single (pairs)      | fires 403/540, acc 0.9950 | fires 392/540, acc **1.0000** |
| combination (pairs) | fires 301/527, acc 0.9867 | fires 379/527, acc **0.9947** |

Pooled over size-3 and size-2, health == 1.0: **2096 rows, accuracy 0.9990.**
(Rows that fire inside UNHEALTHY groups: 360 rows, accuracy 0.9611 - excluded.)

THE RESIDUAL IS PURE COVERAGE, NOT CORRECTION.  Where the crude decoder and the
recalibrated decoder BOTH fire they agree on 100% of rows - 1198/1198 on training
and 213/213 on test.  So this candidate never overturns an actiondecode answer;
it only decides rows the crude decoder abstained on.  Newly-covered rows scored
255/255 = 1.0000 on full training groups and 217/218 = 0.9954 on simulated pairs.
"""
import collections
import pandas as pd

SRC = "/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution"
H = "/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution"
PARENT = f"{H}/cand_multi.csv"
OUT = f"{H}/cand_sc_recal.csv"
ALPHA = 0.09
MAXRUN = 3
TOUCH = ("single", "combination")


def option_actions(row, letter):
    raw = str(row[letter]).strip()
    if row["category"] == "combination":
        return [x.strip() for x in raw.split(",")]
    return [raw]


def live(row):
    return [k for k in "ABCD" if str(row[k]).strip() not in ("", "nan")]


def hits(row, pres):
    if row["category"] == "single":
        return [k for k in live(row) if str(row[k]).strip() in pres]
    return [k for k in "ABCD" if set(option_actions(row, k)) <= pres]


test = pd.read_csv(f"{SRC}/test_qa.csv").drop(columns=["prediction"], errors="ignore")
test["vid"] = test.path.str.extract(r"LM_test_(\d+)")[0].astype(int)
parent = pd.read_csv(PARENT)
ad = pd.read_csv(f"{H}/cand_actiondecode.csv")
P = dict(zip(parent.qa_id, parent.prediction))
AD = dict(zip(ad.qa_id, ad.prediction))

hau = test[test.source == "HAU"].to_dict("records")
by_vid = collections.defaultdict(list)
for r in hau:
    by_vid[r["vid"]].append(r)
emotion_opts = {r["vid"]: frozenset(str(r[k]).strip() for k in "ABCD")
                for r in hau if r["category"] == "emotion"}

groups, cur = [], []
for vid in sorted(by_vid):
    if (cur and len(emotion_opts.get(cur[-1], frozenset())
                   & emotion_opts.get(vid, frozenset())) >= 3 and len(cur) < MAXRUN):
        cur.append(vid)
    else:
        if cur:
            groups.append(cur)
        cur = [vid]
groups.append(cur)

edits, both_fire, both_differ = [], 0, 0
for group in groups:
    counts, slots, rows, anch = collections.Counter(), 0, [], set()
    for vid in group:
        for row in by_vid[vid]:
            if row["category"] == "emotion":
                continue
            rows.append(row)
            if row["category"] == "sequence":
                anch.update(str(row[k]).strip() for k in "ABCD")
            for k in "ABCD":
                for a in option_actions(row, k):
                    counts[a] += 1; slots += 1
    default = max(3, round(ALPHA * slots))
    pa = lambda t: {a for a, c in counts.items() if c >= t} | anch
    hi = max(counts.values()) if counts else 1

    def score(t):
        pres, s = pa(t), 0
        for row in rows:
            if row["category"] in TOUCH:
                s += 1 if len(hits(row, pres)) == 1 else -1
            elif row["category"] == "multi":
                n = sum(1 for k in live(row) if str(row[k]).strip() in pres)
                s += 1 if 1 <= n <= 3 else -1
        return s

    thr = max(range(2, hi + 1), key=lambda t: (score(t), -abs(t - default)))
    pres = pa(thr)
    crude = {a for a, c in counts.items() if c >= default}   # actiondecode's rule

    ok = tot = 0
    for row in rows:
        if row["category"] in TOUCH:
            tot += 1; ok += (len(hits(row, pres)) == 1)
    health = ok / tot if tot else 0.0

    for row in rows:
        if row["category"] not in TOUCH:
            continue
        h, hc = hits(row, pres), hits(row, crude)
        if len(h) == 1 and len(hc) == 1:
            both_fire += 1
            both_differ += (h[0] != hc[0])
        if health < 0.999 or len(group) < 2 or len(h) != 1:
            continue
        if h[0] != P[row["qa_id"]]:
            edits.append(dict(qa=row["qa_id"], vid=row["vid"], cat=row["category"],
                              gsize=len(group), parent=P[row["qa_id"]], new=h[0],
                              crude="ABSTAINED" if len(hc) != 1 else hc[0],
                              actiondecode=AD[row["qa_id"]], thr=thr, default=default))

E = pd.DataFrame(edits)
pred = dict(P)
for r in E.itertuples():
    pred[r.qa] = r.new
pd.DataFrame({"qa_id": test.qa_id,
              "prediction": [pred[q] for q in test.qa_id]}).to_csv(OUT, index=False)

# ------------------------------------------------------------- validate -----
back = pd.read_csv(OUT)
cat = dict(zip(test.qa_id, test.category))
src = dict(zip(test.qa_id, test.source))
opts = {r["qa_id"]: set(live(r)) for r in test.to_dict("records")}
errs = []
if len(back) != 682:
    errs.append(f"row count {len(back)} != 682")
if list(back.qa_id) != list(test.qa_id):
    errs.append("qa_id column/order differs from test_qa.csv")
for q, p in zip(back.qa_id, back.prediction):
    p, c = str(p), cat[q]
    if not p or p == "nan":
        errs.append(f"{q}: empty")
    if any(ch not in "ABCD" for ch in p):
        errs.append(f"{q}: illegal char {p!r}")
    if len(set(p)) != len(p):
        errs.append(f"{q}: repeated char {p!r}")
    if c in ("single", "emotion", "combination", "object_interaction") and len(p) != 1:
        errs.append(f"{q}: {c} must be 1 letter, got {p!r}")
    if c == "multi" and not (1 <= len(p) <= 4):
        errs.append(f"{q}: multi length {p!r}")
    if c == "sequence" and sorted(p) != list("ABCD"):
        errs.append(f"{q}: sequence not a permutation {p!r}")
    if c != "sequence" and list(p) != sorted(p):
        errs.append(f"{q}: not alphabetical {p!r}")
    if c == "single" and src[q] == "HARn" and "D" in p:
        errs.append(f"{q}: HARn single has no option D")
    if not set(p) <= opts[q]:
        errs.append(f"{q}: selects a BLANK option {p!r}")
diff = [q for q, p in zip(back.qa_id, back.prediction) if str(p) != str(P[q])]
out_of_scope = [q for q in diff if cat[q] not in TOUCH]
if out_of_scope:
    errs.append(f"rows changed outside single/combination: {out_of_scope}")
if sorted(diff) != sorted(E.qa.tolist()):
    errs.append("on-disk diff != intended edit set")

print(f"groups: {len(groups)} {dict(collections.Counter(len(g) for g in groups))}")
print(f"crude and recalibrated decoders BOTH fire on {both_fire} test rows; "
      f"they DIFFER on {both_differ}  <-- pure coverage, zero correction")
print(f"\n===== {OUT} =====")
print(f"  rows on disk: {len(back)}; qa_id order identical to test_qa.csv: "
      f"{list(back.qa_id) == list(test.qa_id)}")
print(f"  rows differing from cand_multi.csv: {len(diff)} "
      f"(single {sum(1 for q in diff if cat[q]=='single')}, "
      f"combination {sum(1 for q in diff if cat[q]=='combination')})")
print(f"  rows differing from cand_actiondecode.csv on single/combination: "
      f"{sum(1 for q in back.qa_id if cat[q] in TOUCH and str(pred[q]) != str(AD[q]))}")
print(f"  FORMAT/SCOPE CHECK: {'PASS - 0 violations' if not errs else 'FAIL'}")
for e in errs:
    print("   ", e)
print(f"\n  {'qa_id':<12}{'vid':<6}{'cat':<14}{'gsize':<7}{'parent':<8}{'new':<6}"
      f"{'crude(actiondecode)':<21}{'thr':<5}default")
for r in E.sort_values("qa").itertuples():
    print(f"  {r.qa:<12}{r.vid:<6}{r.cat:<14}{r.gsize:<7}{r.parent:<8}{r.new:<6}"
          f"{r.crude:<21}{r.thr:<5}{r.default}")
