#!/usr/bin/env python3
"""Fit + honestly evaluate the BASE that `inference.sh` ships.

  python3 fit_base.py fit    -> writes cuhk_final/checkpoints/base_v1.{npz,json}
  python3 fit_base.py loo    -> GroupKFold-by-SUBJECT score of the SAME decode path
  python3 fit_base.py loo --nosensor   -> the sensor-free degradation, measured

Everything is refit inside each LOO fold: the 40-class skeleton classifier, the presence
model, the object/emotion/precedence priors. Nothing crosses the subject boundary.
The competition split is subject-disjoint (train users 1-9, 16-24; test 10-11, 25-26),
so a subject-held-out number is the only honest proxy for the leaderboard.
"""
import collections
import csv
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # cuhk_final/
sys.path.insert(0, os.path.join(ROOT, 'multimodal', 'build'))
import basemodel as BM                                                   # noqa: E402

PROJ = os.path.dirname(ROOT) + '/'
DATA = PROJ + 'zero_cost_pipeline/cuhkx_large_data/'
MM = ROOT + '/multimodal/'
SEQ = PROJ + 'historical_seq_workspace/cuhk_large_seq/'
CKPT = ROOT + '/checkpoints/'

TEST_MIX = {'single': 195, 'multi': 144, 'emotion': 144,
            'combination': 139, 'sequence': 39, 'object_interaction': 21}


def load_train():
    rows = list(csv.DictReader(open(DATA + 'training_qa.csv')))
    idk = [k for k in rows[0] if 'qa_id' in k][0]
    for r in rows:
        r['qa_id'] = r[idk]
        r['unit'] = 'Training/' + r['path']
        r['user'] = (r['path'].split('/')[1] if r['source'] == 'HAU'
                     else r['path'].split('/')[2])
    return rows


def presence_labels(rows):
    """Presence/absence per clip, recovered from the labelled TRAIN QA grammar.
    3090 present + 4001 absent over 809 clips with ZERO conflicts (verified)."""
    lab = collections.defaultdict(dict)
    for r in rows:
        u, c = r['unit'], r['category']
        if c == 'single':
            for k in 'ABCD':
                if r[k]:
                    for a in BM.acts(r[k]):
                        lab[u][a] = 1 if k == r['answer'] else 0
        elif c == 'multi':
            for k in 'ABCD':
                if r[k]:
                    for a in BM.acts(r[k]):
                        lab[u][a] = 1 if k in r['answer'] else 0
        elif c == 'combination':
            for k in 'ABCD':
                if r[k] and k == r['answer']:
                    for a in BM.acts(r[k]):
                        lab[u][a] = 1
            for k in 'ABCD':
                if r[k] and k != r['answer']:
                    for a in BM.acts(r[k]):
                        lab[u].setdefault(a, 0)
        elif c == 'sequence':
            for k in 'ABCD':
                if r[k]:
                    for a in BM.acts(r[k]):
                        lab[u][a] = 1
    return lab


def build(qa, by_unit, hold=(), use_sensor=True):
    """Fit every component on the subjects NOT in `hold`. Returns (ck, sens)."""
    harn = np.load(SEQ + 'harn_skel.npz', allow_pickle=True)
    Xh, yh, gh, uh, okh = harn['X'], harn['y'], harn['grp'], harn['units'], harn['ok']
    hau = np.load(MM + 'hau_seg6.npz', allow_pickle=True)
    Xa, ua = hau['X'], hau['units']
    pz = np.load(MM + 'hau_presence.npz', allow_pickle=True)
    classes, canon = pz['classes'].tolist(), pz['canon'].tolist()
    canon_idx = {c: i for i, c in enumerate(canon)}
    gerund = json.load(open(MM + 'harn_single_vocab.json'))
    nframes = json.load(open(MM + 'nframes.json'))

    tr = np.array([g not in hold for g in gh.tolist()]) & okh
    csc = StandardScaler().fit(Xh[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(csc.transform(Xh[tr]), yh[tr])
    order = [list(clf.classes_).index(c) for c in classes]
    ccoef = clf.coef_[order]
    cint = clf.intercept_[order]

    # skeleton latent for every HAU clip (segment posteriors -> pmax + time centroid)
    F = Xa.reshape(-1, Xa.shape[2])
    P = clf.predict_proba(csc.transform(F))[:, order].reshape(
        len(ua), Xa.shape[1], len(classes))
    pres = P.max(1)
    tau = (P / P.sum(1, keepdims=True) * np.arange(P.shape[1])[:, None]).sum(1)
    sens = {u: (pres[i] if use_sensor else None, tau[i] if use_sensor else None)
            for i, u in enumerate(ua.tolist())}
    hpost = {u: BM.clf_posterior(Xh[i], csc.mean_, csc.scale_, ccoef, cint)
             for i, u in enumerate(uh.tolist()) if okh[i]}

    gfreq = collections.Counter()
    for r in qa:
        for k in 'ABCD':
            for a in BM.acts(r[k]):
                gfreq[a] += 1
    gtot = sum(gfreq.values())

    tr_qa = [r for r in qa if r['user'] not in hold]
    lab = presence_labels(tr_qa)

    # ---- presence model ----
    XX, YY = [], []
    for u, d in lab.items():
        rs = by_unit.get(u)
        if not rs or rs[0]['source'] != 'HAU':
            continue
        sf = BM.struct_feats(rs, gfreq, gtot)
        pv = sens.get(u, (None, None))[0]
        nf = nframes.get(u)
        for a, y in d.items():
            if a in sf:
                XX.append(sf[a] + BM.sensor_block(pv, a, canon_idx, nf))
                YY.append(y)
    XX = np.array(XX, np.float32)
    YY = np.array(YY)
    psc = StandardScaler().fit(XX)
    pm = LogisticRegression(max_iter=1000, C=1.0).fit(psc.transform(XX), YY)

    # ---- priors ----
    objmap = collections.defaultdict(collections.Counter)
    for r in tr_qa:
        if r['category'] == 'object_interaction':
            objmap[r['path'].split('/')[1]][r[r['answer']]] += 1
    objmap = {k: v.most_common(1)[0][0] for k, v in objmap.items()}

    slot = collections.defaultdict(lambda: [1., 1., 1.])
    drank = collections.defaultdict(lambda: [1., 1., 1.])
    marg = collections.Counter()
    dur = collections.defaultdict(list)
    g3 = collections.defaultdict(list)
    for r in tr_qa:
        if r['category'] != 'emotion':
            continue
        v = r[r['answer']]
        slot[v][int(r['path'].split('-')[-1]) - 1] += 1
        marg[v] += 1
        n = nframes.get(r['unit'])
        if n:
            dur[v].append(np.log(max(n, 1)))
        g3[r['path'].rsplit('-', 1)[0]].append(r)
    for g, rs in g3.items():
        ns = [nframes.get(r['unit'], -1) for r in rs]
        if len(rs) != 3 or min(ns) < 0 or len(set(ns)) < 3:
            continue
        for r, k in zip(rs, np.argsort(np.argsort(ns))):
            drank[r[r['answer']]][k] += 1
    emo_durgauss = {v: [float(np.mean(x)), float(np.std(x))]
                    for v, x in dur.items() if len(x) >= 5}

    prec = collections.defaultdict(lambda: [1., 1.])
    for r in tr_qa:
        if r['category'] != 'sequence':
            continue
        ordr = [r[k] for k in r['answer'] if r.get(k)]
        for i1 in range(len(ordr)):
            for j1 in range(i1 + 1, len(ordr)):
                a, b = ordr[i1], ordr[j1]
                key = a + '\t' + b if a < b else b + '\t' + a
                prec[key][0 if a < b else 1] += 1

    ck = dict(
        classes=classes, canon=canon, canon_idx=canon_idx, gerund=gerund,
        objmap=objmap, slot=dict(slot), drank=dict(drank), prec=dict(prec),
        emo_marginal=dict(marg), emo_durgauss=emo_durgauss,
        gfreq=dict(gfreq), gtot=gtot,
        cmean=csc.mean_, cscale=csc.scale_, ccoef=ccoef, cint=cint,
        pmean=psc.mean_, pscale=psc.scale_,
        pcoef=pm.coef_[0], pint=float(pm.intercept_[0]),
        use_sensor=use_sensor,
    )
    return ck, sens, hpost, nframes


def decode_all(units, by_unit, ck, sens, hpost, nframes, te_dr=None, train_slot=False):
    pred = {}
    for u in units:
        rs = by_unit[u]
        pv, tv = sens.get(u, (None, None))
        nf = nframes.get(u)
        pr = {}
        if rs[0]['source'] == 'HAU':
            sf = BM.struct_feats(rs, ck['gfreq'], ck['gtot'])
            if sf:
                aa = sorted(sf)
                pr = BM.presence_probs(aa, sf, pv, ck['canon_idx'], nf,
                                       ck['pmean'], ck['pscale'], ck['pcoef'], ck['pint'])
        slot_no = None
        if train_slot:
            try:
                slot_no = int(rs[0]['path'].split('-')[-1]) - 1
            except Exception:
                slot_no = None
        pred.update(BM.decode_unit(rs, pr, hpost.get(u), tv, ck,
                                   dur_rank=(te_dr or {}).get(u), slot_no=slot_no,
                                   nframes=nf))
    return pred


def report(qa, pred, tag):
    per = collections.defaultdict(lambda: [0, 0])
    pu = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in qa:
        if r['qa_id'] not in pred:
            continue
        ok = pred[r['qa_id']] == r['answer']
        per[r['category']][0] += ok
        per[r['category']][1] += 1
        pu[r['user']][r['category']][0] += ok
        pu[r['user']][r['category']][1] += 1
    print(f'\n=== BASE {tag}  GroupKFold-by-SUBJECT ===')
    tot = 0.
    for c, n in TEST_MIX.items():
        k, m = per[c]
        if not m:
            continue
        tot += n * k / m
        print(f'  {c:20s} {k}/{m} = {k/m:.4f}   (test rows {n})')
    score = tot / sum(TEST_MIX.values())
    print(f'  -> TEST-MIX WEIGHTED BASE SCORE = {score:.4f}')
    vals = [sum(TEST_MIX[c] * pu[u][c][0] / pu[u][c][1] for c in TEST_MIX if pu[u][c][1]) /
            sum(TEST_MIX[c] for c in TEST_MIX if pu[u][c][1]) for u in pu]
    print(f'  per-subject: mean {np.mean(vals):.4f} sd {np.std(vals):.4f} '
          f'min {min(vals):.4f} max {max(vals):.4f}  n={len(vals)}')
    return score


def e2e(qa, by_unit, use_sensor, t0):
    """THE NUMBER THE PIPELINE ACTUALLY PRODUCES: base + the four structural decoders,
    in `run_inference.py`'s exact order, LOO-by-SUBJECT with everything refit per fold.

    Caveat, stated rather than hidden: on TRAIN the emotion triplets are contiguous and
    perfectly recoverable from the path, while TEST paths are anonymised (measured
    ceiling 93 of 144 emotion rows recoverable). So the emotion arm here is OPTIMISTIC
    by construction; every other arm is like-for-like.
    """
    sys.path.insert(0, os.path.join(ROOT, 'structural_decoder'))
    import emotion_triplet, action_decoder, multi_decoder, abstention_recovery

    nframes = json.load(open(MM + 'nframes.json'))
    users = sorted({r['user'] for r in qa}, key=lambda s: int(s[4:]))
    folds = [users[i::6] for i in range(6)]
    pred, cov = {}, collections.Counter()
    for fi, fold in enumerate(folds):
        hold = set(fold)
        ck, sens, hpost, nf = build(qa, by_unit, hold=hold, use_sensor=use_sensor)
        te = [r for r in qa if r['user'] in hold]
        tr = [r for r in qa if r['user'] not in hold]
        units = {r['unit'] for r in te}
        p = {r['qa_id']: 'A' for r in te}
        if '--nobase' not in sys.argv:
            base = decode_all(units, by_unit, ck, sens, hpost, nf)
            p.update(base)
            cov['base'] += len(base)
        tbl = emotion_triplet.fit_slot_table(tr)
        dtab = emotion_triplet.fit_duration_table(tr, nframes)
        for name, fn in (
                ('emotion', lambda: emotion_triplet.decode_with_duration(
                    te, tbl, nframes, dtab)),
                ('action', lambda: action_decoder.decode(te)),
                ('multi', lambda: multi_decoder.decode(te)),
                ('abstain', lambda: abstention_recovery.decode(te))):
            try:
                u = fn() or {}
                cov[name] += len(u)
                p.update(u)
            except Exception as e:
                sys.stderr.write(f'  {name} failed: {e}\n')
        pred.update(p)
        sys.stderr.write(f'  fold {fi} {time.time()-t0:.0f}s\n')
    s = report(qa, pred, f"END-TO-END {'NO BASE (old pipeline)' if '--nobase' in sys.argv else 'base+4 decoders'} sensor={use_sensor}")
    print('  decoder coverage (rows decided, all folds):', dict(cov))
    print(f'  wall {time.time()-t0:.0f}s')
    return s


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'loo'
    use_sensor = '--nosensor' not in sys.argv
    train_slot = '--trainslot' in sys.argv
    t0 = time.time()
    qa = load_train()
    by_unit = collections.defaultdict(list)
    for r in qa:
        by_unit[r['unit']].append(r)

    if mode == 'fit':
        ck, _, _, _ = build(qa, by_unit, hold=(), use_sensor=True)
        os.makedirs(CKPT, exist_ok=True)
        arrays = {k: np.asarray(ck[k]) for k in
                  ('cmean', 'cscale', 'ccoef', 'cint', 'pmean', 'pscale', 'pcoef')}
        maps = {k: ck[k] for k in
                ('classes', 'canon', 'gerund', 'objmap', 'slot', 'drank', 'prec',
                 'emo_marginal', 'emo_durgauss', 'gfreq', 'gtot')}
        maps['pint'] = ck['pint']
        BM.save_checkpoint(CKPT + 'base_v1.npz', CKPT + 'base_v1.json', arrays, maps)
        print(f'wrote {CKPT}base_v1.npz + base_v1.json  ({time.time()-t0:.0f}s)')
        print('  sizes:', {k: os.path.getsize(CKPT + k)
                           for k in ('base_v1.npz', 'base_v1.json')})
        return

    if mode == 'e2e':
        return e2e(qa, by_unit, use_sensor, t0)

    users = sorted({r['user'] for r in qa}, key=lambda s: int(s[4:]))
    folds = [users[i::6] for i in range(6)]
    pred = {}
    for fi, fold in enumerate(folds):
        ck, sens, hpost, nframes = build(qa, by_unit, hold=set(fold),
                                         use_sensor=use_sensor)
        units = {r['unit'] for r in qa if r['user'] in fold}
        pred.update(decode_all(units, by_unit, ck, sens, hpost, nframes,
                               train_slot=train_slot))
        sys.stderr.write(f'  fold {fi} {time.time()-t0:.0f}s\n')
    tag = f'sensor={use_sensor} trainslot={train_slot}'
    report(qa, pred, tag)
    print(f'  wall {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
