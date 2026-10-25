"""Modality detection + graceful degradation for the live organizer reproduction.

CONTRACT: `./inference.sh DATA_DIR` -> predictions CSV. We do NOT know whether the live
DATA_DIR carries the non-visual supplement. Nothing here may raise on a missing modality;
every probe returns a boolean and the pipeline drops to the next tier down.

TIERS (each strictly additive; a lower tier is always a valid, complete submission)
  T0 STRUCTURAL   needs only test_qa.csv                     -> always available
  T1 DURATION     needs a per-clip frame/duration count      -> emotion pos+dur  (our +12)
  T2 SKELETON     needs Skeleton/predictions/*.json contents -> the fused BASE latent:
                  presence, the sequence time-centroid and the HARn 40-class arm.
                  Measured footprint 2026-09-09: 81 of 682 rows change when skeleton is
                  present vs absent (single 32, sequence 21, multi 13, obj 11, comb 4).

IMU and RADAR are probed and reported in `caps` as a factual inventory of what the
DATA_DIR contains, but they are NOT tiers: no code consumes them.  `plan()` therefore
does not announce them.  See the note in plan().

T1 IS THE HIGH-VALUE, LOW-DEPENDENCY TIER. Duration can be recovered THREE ways, in
order of preference, so it survives almost any DATA_DIR shape:
  (a) count Skeleton/predictions/*.json  (one JSON per video frame - exact, no decode)
  (b) ffprobe the Depth mp4              (nb_frames / duration - the supplement README
      states all modalities share the same time window as the video clip)
  (c) cv2 CAP_PROP_FRAME_COUNT on the mp4
  (d) mp4_frames.frame_count - PURE STDLIB atom walk of the sample table, no decoding and
      no third-party package. ffprobe AND cv2 are both absent on this machine, so (d) is
      what actually keeps the +12-row emotion mechanism alive on a Depth-only DATA_DIR.
Only if all three fail does emotion fall back to the structural positional arm.
"""
import os, subprocess, glob, json


def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def find_unit_dir(data_dir, unit_rel):
    """unit_rel e.g. 'large_model_track_test/LM_test_0066'. Tolerates an extra
    Testing/ level, a flat layout, or the supplement being unpacked alongside."""
    cands = [os.path.join(data_dir, unit_rel),
             os.path.join(data_dir, 'Testing', unit_rel),
             os.path.join(data_dir, os.path.basename(unit_rel))]
    for root, _, _ in os.walk(data_dir):
        if os.path.basename(root) == os.path.basename(unit_rel):
            cands.append(root)
            break
    return _first(cands)


def probe(data_dir, sample_units):
    """-> dict of booleans + a per-unit duration source. Never raises."""
    caps = {'qa': False, 'duration': False, 'duration_via': None,
            'skeleton': False, 'imu': False, 'radar': False, 'depth': False}
    try:
        caps['qa'] = bool(_first([os.path.join(data_dir, 'test_qa.csv'),
                                  os.path.join(data_dir, 'Testing', 'test_qa.csv')]))
        hits = {'skeleton': 0, 'imu': 0, 'radar': 0, 'depth': 0}
        n = 0
        for u in sample_units[:12]:
            d = find_unit_dir(data_dir, u)
            if not d:
                continue
            n += 1
            if glob.glob(os.path.join(d, 'Skeleton', 'predictions', '*.json')):
                hits['skeleton'] += 1
            if glob.glob(os.path.join(d, 'IMU', '*.csv')):
                hits['imu'] += 1
            if os.path.exists(os.path.join(d, 'Radar', 'Radar.csv')):
                hits['radar'] += 1
            if glob.glob(os.path.join(d, 'Depth', '*.mp4')):
                hits['depth'] += 1
        if n:
            for k in hits:
                caps[k] = hits[k] >= max(1, n // 2)
        if caps['skeleton']:
            caps['duration'], caps['duration_via'] = True, 'skeleton_json_count'
        elif caps['depth']:
            # mp4_frames is pure stdlib and always importable, so duration NEVER depends
            # on ffprobe/cv2 being installed on the organizer's machine.
            caps['duration'] = True
            caps['duration_via'] = ('ffprobe' if _have('ffprobe')
                                    else 'cv2' if _have_cv2() else 'mp4_atom')
    except Exception as ex:                       # never let a probe kill the run
        caps['probe_error'] = repr(ex)
    return caps


def _have(binname):
    try:
        subprocess.run([binname, '-version'], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def _have_cv2():
    try:
        import cv2  # noqa
        return True
    except Exception:
        return False


def unit_duration(data_dir, unit_rel, via):
    """Frames for one clip, or None. Never raises."""
    try:
        d = find_unit_dir(data_dir, unit_rel)
        if not d:
            return None
        if via == 'skeleton_json_count':
            n = len(glob.glob(os.path.join(d, 'Skeleton', 'predictions', '*.json')))
            return n or None
        mp4 = _first(sorted(glob.glob(os.path.join(d, 'Depth', '*.mp4'))))
        if not mp4:
            return None
        if via == 'mp4_atom':
            from mp4_frames import frame_count
            return frame_count(mp4)
        if via == 'ffprobe':
            out = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                 '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', mp4],
                capture_output=True, text=True, timeout=120).stdout.strip()
            return int(out) if out.isdigit() else None
        import cv2
        cap = cv2.VideoCapture(mp4)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n or None
    except Exception:
        return None


def plan(caps):
    tiers = ['T0_structural']
    if caps.get('duration'):
        tiers.append('T1_duration_emotion')
    if caps.get('skeleton'):
        tiers.append('T2_skeleton')
    # IMU and RADAR are DETECTED (see caps) but deliberately NOT announced as tiers:
    # nothing in the pipeline consumes them.  Measured 2026-09-09: with IMU and Radar
    # both present, skeleton-removed vs QA-only changes 0 of 682 rows.  A tier line is
    # a capability claim the organizer's verifier reads, so we only announce tiers that
    # have a consumer.  Do not add T3/T4 back without a measured n_changed_behavior > 0.
    return tiers


if __name__ == '__main__':
    import sys
    dd = sys.argv[1] if len(sys.argv) > 1 else '.'
    su = sys.argv[2:] or ['large_model_track_test/LM_test_0001']
    c = probe(dd, su)
    print(json.dumps(c, indent=2))
    print('ENABLED TIERS:', ' -> '.join(plan(c)))
