"""Inference-time BASE: run the shipped checkpoint over a live DATA_DIR.

`run_inference.py` calls `predict()` ONCE, before the four structural decoders, so every
row of the submission carries a real model prediction instead of the literal 'A' default.
The decoders then overwrite the rows they fire on — they are LB-confirmed and must win.

NOTHING HERE REFITS. The 40-class skeleton classifier, the presence model and all the
priors come from `checkpoints/base_v1.{npz,json}`, fitted offline by `code/fit_base.py`.

GRACEFUL DEGRADATION, in order:
  * no checkpoint            -> predict() returns {}; the pipeline is exactly as before
  * checkpoint, no skeleton  -> structural-only latent (sensor sentinel features)
  * checkpoint + skeleton    -> full fused latent + the HARn 40-class arm
Every stage is wrapped; a failure in one clip cannot affect another.
"""
import collections
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import basemodel as BM                                                   # noqa: E402

NSEG = 6                                            # must match hau_seg6.npz

# Observed-behaviour counters for the caller's tier report.  The entry point derives its
# 'tier plan (effective)' from these rather than from the probe, so the line it prints
# describes what the run DID, not what the DATA_DIR appeared to contain.
LAST_SKELETON_CLIPS = 0
LAST_HARN_CLIPS = 0


def _log(m):
    print(f'[base] {m}', flush=True)


# --------------------------------------------------------------------- skeleton input
def _unit_dir(data_dir, unit_rel, cache={}):
    if unit_rel in cache:
        return cache[unit_rel]
    d = None
    for c in (os.path.join(data_dir, unit_rel),
              os.path.join(data_dir, 'Testing', unit_rel),
              os.path.join(data_dir, os.path.basename(unit_rel))):
        if os.path.isdir(c):
            d = c
            break
    cache[unit_rel] = d
    return d


def _load_kp(unit_dir):
    """(T,17,3) keypoints for a clip from Skeleton/predictions/*.json, or None."""
    import skel_feat
    files = sorted(glob.glob(os.path.join(unit_dir, 'Skeleton', 'predictions', '*.json')))
    if len(files) < 4:
        return None
    frames = []
    for p in files:
        try:
            with open(p, 'rb') as fh:
                kp = skel_feat.parse_frame(fh.read())
        except Exception:
            kp = None
        if kp is not None:
            frames.append(kp)
    if len(frames) < 2:
        return None
    return np.stack(frames, 0)


def _clip_sensor(kp, ck):
    """-> (pres 40-vec, tau 40-vec, whole-clip 40-vec posterior). Any may be None."""
    import skel_feat
    cm, cs, cc, ci = ck['cmean'], ck['cscale'], ck['ccoef'], ck['cint']
    whole = BM.clf_posterior(skel_feat.feat_from_kp(kp), cm, cs, cc, ci)
    if kp.shape[0] < NSEG * 2:
        return None, None, whole
    segs = np.array_split(np.arange(kp.shape[0]), NSEG)
    P = np.stack([BM.clf_posterior(skel_feat.feat_from_kp(kp[ix]), cm, cs, cc, ci)
                  for ix in segs if len(ix) >= 2])
    if len(P) < 2:
        return None, None, whole
    pres = P.max(0)
    tau = (P / P.sum(0, keepdims=True) * np.arange(len(P))[:, None]).sum(0)
    return pres, tau, whole


# --------------------------------------------------------------------------- entry
def predict(rows, data_dir, ckpt_dir, durations=None, video_key=lambda p: p):
    """-> {qa_id: prediction} for EVERY row we can answer. Never raises."""
    npz = os.path.join(ckpt_dir, 'base_v1.npz')
    js = os.path.join(ckpt_dir, 'base_v1.json')
    if not (os.path.isfile(npz) and os.path.isfile(js)):
        _log('checkpoint base_v1 missing; base disabled (structural decoders only)')
        return {}
    try:
        ck = BM.load_checkpoint(npz, js)
        ck['pint'] = float(ck['pint'])
    except Exception as e:
        _log(f'checkpoint unreadable ({type(e).__name__}: {e}); base disabled')
        return {}

    # A shipped duration index is keyed 'Testing/large_model_track_test/LM_test_0031'
    # while a QA path yields 'large_model_track_test/LM_test_0031'. Index by SUFFIX so
    # the two cannot silently miss each other (the same class of bug that once left the
    # emotion duration table fitted to zero adverbs).
    durations = dict(durations or {})
    for k, v in list(durations.items()):
        parts = [p for p in str(k).split('/') if p]
        for n in (3, 2):
            if len(parts) >= n:
                durations.setdefault('/'.join(parts[-n:]), v)

    by_unit = collections.defaultdict(list)
    for r in rows:
        by_unit[video_key(str(r.get('path') or ''))].append(r)

    global LAST_SKELETON_CLIPS, LAST_HARN_CLIPS
    LAST_SKELETON_CLIPS = LAST_HARN_CLIPS = 0
    pred, n_sens, n_har = {}, 0, 0
    for unit, rs in by_unit.items():
        pres = tau = whole = None
        try:
            d = _unit_dir(data_dir, unit) if unit else None
            if d:
                kp = _load_kp(d)
                if kp is not None:
                    pres, tau, whole = _clip_sensor(kp, ck)
                    n_sens += 1
                    if durations.get(unit) is None:
                        durations[unit] = int(kp.shape[0])
        except Exception:
            pres = tau = whole = None
        nf = durations.get(unit)
        try:
            sf = BM.struct_feats(rs, ck['gfreq'], ck['gtot'])
            pr = {}
            if sf:
                aa = sorted(sf)
                pr = BM.presence_probs(aa, sf, pres, ck['canon_idx'], nf,
                                       ck['pmean'], ck['pscale'], ck['pcoef'], ck['pint'])
            hp = whole if any(r.get('source') == 'HARn' for r in rs) else None
            if hp is not None:
                n_har += 1
            pred.update(BM.decode_unit(rs, pr, hp, tau, ck, nframes=nf))
        except Exception:
            continue
    LAST_SKELETON_CLIPS, LAST_HARN_CLIPS = n_sens, n_har
    _log(f'{len(by_unit)} clips; skeleton on {n_sens}; HARn arm on {n_har}; '
         f'{len(pred)} of {len(rows)} rows predicted')
    return pred
