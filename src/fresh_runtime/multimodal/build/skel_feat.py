"""Skeleton motion feature extraction from LMT Skeleton/predictions JSON frames."""
import json
import numpy as np

NJ = 17  # keypoints


def parse_frame(raw):
    try:
        d = json.loads(raw)
    except Exception:
        return None
    if not d:
        return None
    # multiple detections possible -> take the one with highest summed confidence
    best = max(d, key=lambda person: sum(person.get('keypoint_scores', [0]) or [0]))
    kp = best.get('keypoints')
    if kp is None or len(kp) != NJ:
        return None
    arr = np.asarray(kp, dtype=np.float32)
    if arr.shape != (NJ, 3):
        return None
    return arr


def load_kp_seq(z, paths):
    """paths: ordered list of json zip-entries -> (T,17,3) float32 array (T = frames that parsed ok)."""
    frames = []
    for p in paths:
        try:
            raw = z.read(p)
        except KeyError:
            continue
        kp = parse_frame(raw)
        if kp is not None:
            frames.append(kp)
    if len(frames) < 2:
        return None
    return np.stack(frames, axis=0)


FDIM = 196


def feat_from_kp(kp):
    """(T,17,3) -> fixed-length motion feature vector."""
    T = kp.shape[0]
    center = kp.mean(axis=1, keepdims=True)          # (T,1,3) per-frame centroid
    c = kp - center                                    # translation-normalized
    scale = c.reshape(T, -1).std(axis=1).mean() + 1e-6  # crude body/camera-scale normalizer
    cn = c / scale

    if T >= 2:
        vel = np.diff(cn, axis=0)                      # (T-1,17,3)
        velmag = np.linalg.norm(vel, axis=2)            # (T-1,17)
    else:
        vel = np.zeros((0, NJ, 3), dtype=np.float32)
        velmag = np.zeros((0, NJ), dtype=np.float32)

    if T >= 3:
        acc = np.diff(vel, axis=0)
        accmag = np.linalg.norm(acc, axis=2)
    else:
        accmag = np.zeros((0, NJ), dtype=np.float32)

    feats = [cn.mean(axis=0).ravel(), cn.std(axis=0).ravel()]  # 51 + 51

    if len(velmag):
        feats += [velmag.mean(axis=0), velmag.std(axis=0), velmag.max(axis=0)]  # 17*3
    else:
        feats += [np.zeros(NJ, dtype=np.float32)] * 3

    if len(accmag):
        feats += [accmag.mean(axis=0), accmag.max(axis=0)]  # 17*2
    else:
        feats += [np.zeros(NJ, dtype=np.float32)] * 2

    if len(velmag):
        energy = velmag.sum(axis=1)
        feats.append(np.array([energy.mean(), energy.std(), energy.max(),
                                np.percentile(energy, 90)], dtype=np.float32))
    else:
        feats.append(np.zeros(4, dtype=np.float32))

    ext = cn.reshape(T, -1, 3)
    bbox_ext = (ext.max(axis=(0, 1)) - ext.min(axis=(0, 1)))  # 3
    feats.append(bbox_ext)
    feats.append(np.array([T, scale], dtype=np.float32))

    out = np.concatenate([np.nan_to_num(x).astype(np.float32) for x in feats])
    assert out.shape[0] == FDIM, out.shape
    return out


def unit_feat(z, paths):
    kp = load_kp_seq(z, paths)
    if kp is None:
        return None
    return feat_from_kp(kp)


def segment_feats(z, paths, nseg=4):
    """Split an ordered frame-path list into nseg contiguous chunks and featurize each.
    Returns list of length nseg of (FDIM,) vectors, or None if too few usable frames."""
    kp = load_kp_seq(z, paths)
    if kp is None or kp.shape[0] < nseg * 2:
        return None
    chunks = np.array_split(np.arange(kp.shape[0]), nseg)
    out = []
    for idxs in chunks:
        if len(idxs) < 2:
            return None
        out.append(feat_from_kp(kp[idxs]))
    return out
