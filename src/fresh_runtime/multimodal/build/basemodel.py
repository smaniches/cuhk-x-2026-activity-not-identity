"""BASE MODEL — the prediction the runnable pipeline OWNS for every row.

Before this module, `run_inference.py` seeded every row with the literal letter 'A' and
let four structural decoders overwrite ~101 of 682 rows. The other 581 rows were a
default. This module supplies a real prediction for all of them, from ONE latent per
clip: P(action present) over the 40-action vocabulary (+ a per-action time centroid),
read out into all six question types.

THE SAME FUNCTIONS ARE USED BY THE FITTER, THE LOO EVALUATOR, AND `inference.sh`.
That is deliberate: the honest cross-subject number and the number the pipeline produces
come from one code path, so they cannot drift.

FEATURES per (clip, candidate action)
  structural (8) — from OPTION TEXT ONLY, so always available:
    times the action occurs across the clip's options, in how many of the clip's
    questions, whether it is an option of the `sequence` question (present by grammar),
    its share of the clip's combination options, its share of its multi options,
    whether it is a `single` option, its global corpus option-frequency, and the clip's
    candidate-set size.
  sensor (4) — from the skeleton stream, ABSENT-TOLERANT (sentinel when missing):
    log max-over-6-segments 40-class posterior for the action, its within-clip z-score,
    its within-clip rank, and log clip duration.

DEGRADATION. `sensor_block()` returns the sentinel vector whenever the skeleton is
missing, so the identical model runs with or without sensors — it simply loses the
sensor evidence. No branch, no crash, no separate code path.
"""
import collections
import itertools
import json

import numpy as np

N_STRUCT = 8
N_SENSOR = 4
SENSOR_ABSENT = [-9.0, -9.0, 0.0, 0.0]


def acts(t):
    return [a.strip() for a in t.split(',')] if t else []


# --------------------------------------------------------------------------- structural
def struct_feats(unit_rows, gfreq, gtot):
    """rows of ONE clip -> {action: [8 structural features]}. Option text only."""
    cand = collections.Counter()
    nq = collections.Counter()
    ncomb = collections.Counter()
    nmulti = collections.Counter()
    nsingle = collections.Counter()
    inseq = set()
    comb_tot = multi_tot = 0
    for r in unit_rows:
        if r['category'] == 'emotion':
            continue
        seen = set()
        for k in 'ABCD':
            if not r.get(k):
                continue
            aa = acts(r[k])
            for a in aa:
                cand[a] += 1
                seen.add(a)
            if r['category'] == 'combination':
                for a in set(aa):
                    ncomb[a] += 1
            elif r['category'] == 'multi':
                for a in set(aa):
                    nmulti[a] += 1
            elif r['category'] == 'single':
                for a in set(aa):
                    nsingle[a] += 1
        if r['category'] == 'combination':
            comb_tot += sum(1 for k in 'ABCD' if r.get(k))
        if r['category'] == 'multi':
            multi_tot += sum(1 for k in 'ABCD' if r.get(k))
        if r['category'] == 'sequence':
            inseq |= seen
        for a in seen:
            nq[a] += 1
    out = {}
    for a in cand:
        out[a] = [cand[a], nq[a], 1.0 * (a in inseq),
                  ncomb[a] / max(comb_tot, 1), nmulti[a] / max(multi_tot, 1),
                  float(nsingle[a] > 0),
                  float(np.log(gfreq.get(a, 0) / max(gtot, 1) + 1e-9)),
                  len(cand)]
    return out


# --------------------------------------------------------------------------- sensor
def sensor_block(pres_vec, action, canon_idx, nframes):
    """4 sensor features for one action, or the ABSENT sentinel. Never raises."""
    if pres_vec is None or action not in canon_idx:
        return list(SENSOR_ABSENT)
    try:
        v = np.asarray(pres_vec, dtype=np.float64)
        j = canon_idx[action]
        lv = np.log(v + 1e-9)
        return [float(lv[j]),
                float((lv[j] - lv.mean()) / (lv.std() + 1e-6)),
                float((v > v[j]).sum()),
                float(np.log(max(nframes or 1, 1)))]
    except Exception:
        return list(SENSOR_ABSENT)


def feat_matrix(actions, sf, pres_vec, canon_idx, nframes):
    return np.array([sf[a] + sensor_block(pres_vec, a, canon_idx, nframes)
                     for a in actions], np.float32)


def presence_probs(actions, sf, pres_vec, canon_idx, nframes, pmean, pscale, pcoef, pint):
    """-> {action: P(present)}. Pure numpy, so no sklearn needed at inference."""
    if not actions:
        return {}
    X = feat_matrix(actions, sf, pres_vec, canon_idx, nframes)
    Z = (X - pmean) / pscale
    z = Z @ pcoef + pint
    return dict(zip(actions, 1.0 / (1.0 + np.exp(-z))))


def clf_posterior(x196, cmean, cscale, ccoef, cint):
    """40-class skeleton posterior for one whole-clip feature vector (softmax LR)."""
    z = ((np.asarray(x196, np.float64) - cmean) / cscale) @ ccoef.T + cint
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# --------------------------------------------------------------------------- readouts
def decode_unit(unit_rows, pr, hpost, tau, ck, dur_rank=None, slot_no=None,
                nframes=None, MULTI_P=0.45, SEQ_LAM=1.0):
    """All of ONE clip's answers off the single latent. -> {qa_id: answer}.

    pr      {action: P(present)}                 (may be {})
    hpost   40-vector skeleton posterior or None (HARn arm)
    tau     40-vector time centroid or None      (sequence arm)
    ck      loaded checkpoint dict
    """
    classes = ck['classes']
    canon_idx = ck['canon_idx']
    gerund = ck['gerund']
    objmap = ck['objmap']
    slot = ck['slot']
    drank = ck['drank']
    prec = ck['prec']
    out = {}
    for r in unit_rows:
        cat = r['category']
        opts = {k: r[k] for k in 'ABCD' if r.get(k)}
        if not opts:
            continue
        try:
            if cat == 'emotion':
                # TEST paths are anonymised, so the filming slot and the within-triplet
                # duration rank are usually unknown. Measured fallback (held-out
                # subjects): marginal adverb prior alone 0.3151; prior + an ABSOLUTE
                # log-duration Gaussian 0.3834. Use the latter whenever a duration is
                # known. Where the LB-confirmed emotion_triplet decoder recovers a
                # triplet it overwrites this row anyway.
                marg = ck.get('emo_marginal', {})
                gauss = ck.get('emo_durgauss', {})
                mtot = sum(marg.values()) or 1.0
                ld = float(np.log(max(nframes or 1, 1))) if nframes else None
                s_ = {}
                for k, v in opts.items():
                    sv = slot.get(v, [1., 1., 1.])
                    dv = drank.get(v, [1., 1., 1.])
                    s = 0.0
                    if slot_no is not None:
                        s += float(np.log(sv[slot_no] / sum(sv)))
                    if dur_rank is not None:
                        s += float(np.log(dv[dur_rank] / sum(dv)))
                    if slot_no is None and dur_rank is None:
                        s += float(np.log(marg.get(v, 0.5) / mtot))
                        g = gauss.get(v)
                        if g and ld is not None and ld > 0:
                            mu, sd = g
                            sd = max(sd, 1e-2)
                            s += float(-0.5 * ((ld - mu) / sd) ** 2 - np.log(sd))
                    s_[k] = s
                out[r['qa_id']] = max(s_, key=s_.get)
            elif r.get('source') == 'HARn':
                if hpost is None:
                    continue                      # leave to the structural layer
                m = objmap if cat == 'object_interaction' else gerund
                s_ = {k: sum(hpost[j] for j, c in enumerate(classes) if m.get(c) == v)
                      for k, v in opts.items()}
                out[r['qa_id']] = max(s_, key=s_.get)
            elif cat in ('single', 'combination'):
                def sc_opt(t):
                    aa = acts(t)
                    return (float(np.mean([np.log(pr.get(a, .5) + 1e-9) for a in aa]))
                            if aa else -99.0)
                s_ = {k: sc_opt(v) for k, v in opts.items()}
                out[r['qa_id']] = max(s_, key=s_.get)
            elif cat == 'multi':
                sel = [k for k in sorted(opts) if pr.get(opts[k], 0.) >= MULTI_P]
                if not sel:
                    sel = [max(opts, key=lambda k: pr.get(opts[k], 0.))]
                out[r['qa_id']] = ''.join(sel)
            elif cat == 'sequence':
                ks = sorted(opts)
                best, bs = ''.join(ks), -1e18
                usable = tau is not None and all(opts[k] in canon_idx for k in ks)
                for perm in itertools.permutations(ks):
                    s = 0.0
                    for x in range(len(perm)):
                        for y in range(x + 1, len(perm)):
                            a, b = opts[perm[x]], opts[perm[y]]
                            key = a + '\t' + b if a < b else b + '\t' + a
                            v = prec.get(key, [1., 1.])
                            s += float(np.log(v[0 if a < b else 1] / sum(v)))
                    if usable:
                        t = [tau[canon_idx[opts[k]]] for k in perm]
                        s += SEQ_LAM * sum(1.0 for x in range(len(t))
                                           for y in range(x + 1, len(t)) if t[x] < t[y])
                    if s > bs:
                        bs, best = s, ''.join(perm)
                out[r['qa_id']] = best
        except Exception:
            continue                              # a bad row never kills the clip
    return out


# --------------------------------------------------------------------------- checkpoint
def save_checkpoint(path_npz, path_json, arrays, maps):
    np.savez_compressed(path_npz, **arrays)
    with open(path_json, 'w') as f:
        json.dump(maps, f)


def load_checkpoint(path_npz, path_json):
    d = np.load(path_npz, allow_pickle=False)
    m = json.load(open(path_json))
    ck = {k: d[k] for k in d.files}
    ck.update(m)
    ck['canon_idx'] = {c: i for i, c in enumerate(ck['canon'])}
    return ck
