#!/usr/bin/env python3
"""CUHK-X Large — the single inference entry point.

CONTRACT (organizer live verification):  ./inference.sh DATA_DIR  ->  submission.csv
The SAME pipeline that runs experiments must produce the submission. No magic script.

DESIGN RULE: NEVER CRASH. Modalities present in the live DATA_DIR are unknown, so every
tier is wrapped; a tier that fails degrades to the tier below, and T0 alone is a complete
valid submission. A crash during organizer reproduction is far worse than a lower score.
"""
import argparse, json, os, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "structural_decoder"))
sys.path.insert(0, str(HERE / "multimodal" / "build"))

import common
import emotion_triplet, action_decoder, multi_decoder, abstention_recovery


def log(m): print(f"[inference] {m}", flush=True)


def find_qa(data_dir: Path) -> Path:
    for c in ("test_qa.csv", "test.csv", "qa/test_qa.csv"):
        p = data_dir / c
        if p.is_file():
            return p
    hits = sorted(data_dir.rglob("test_qa.csv"))
    if hits:
        return hits[0]
    raise SystemExit(f"no test_qa.csv under {data_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("-o", "--out", default="submission.csv")
    a = ap.parse_args()
    data_dir = Path(a.data_dir).resolve()
    log(f"DATA_DIR={data_dir}")

    rows = common.read_qa(str(find_qa(data_dir)))
    log(f"loaded {len(rows)} QA rows")
    pred = {r["qa_id"]: (r.get("prediction") or "A") for r in rows}

    # ---- modality tiering (never raises) ----
    tiers, caps = ["T0"], {}
    try:
        import modality_probe
        units = []
        for r in rows[:400]:
            p = r.get("path") or ""
            if p:
                # the CLIP directory, not the modality subdir: 'a/LM_test_1/Depth/x.mp4'
                # -> 'a/LM_test_1'. Using dirname() here pointed the probe at the Depth/
                # folder and it reported every modality absent.
                units.append(common.video_key(p))
        caps = modality_probe.probe(str(data_dir), sorted(set(units))[:12])
        log(f"modality caps={caps}")
    except Exception as e:
        # NOT "T0 structural only": durations can still resolve from the shipped index
        # and T1 can still fire.  What this run actually does is reported by the
        # "tier plan (effective)" line below, after resolution.
        log(f"modality probe unavailable ({type(e).__name__}: {e}); "
            f"continuing with no modality inventory")
    # NOTE: the tier plan is logged AFTER duration resolution, below, so that the line
    # reports what the run WILL DO rather than what the DATA_DIR merely contains.
    # `caps['duration']` describes the DATA_DIR alone and misses the shipped index, so
    # planning here printed "T0 only" on runs where T1 then went on to fire.

    # ---- T0: the four LB-confirmed structural decoders, in their confirmed order ----
    from collections import Counter

    def _load_ckpt(name):
        p = HERE / "checkpoints" / name
        if not p.is_file():
            log(f"  checkpoint MISSING: {name}")
            return None
        raw = json.loads(p.read_text())
        return {w: Counter({int(k): v for k, v in c.items()}) for w, c in raw.items()}

    tbl  = _load_ckpt("emotion_slot_table.json")
    dtab = _load_ckpt("emotion_duration_table.json")

    # Durations: prefer a shipped index; else read them live from mp4 `stsz` atoms
    # (pure stdlib, no ffprobe/cv2 — so a Depth-only DATA_DIR still enables T1).
    # KEY-DRIVEN, NOT FILE-DRIVEN.  The shipped index is keyed to the clips we already
    # hold; on the organizer's FRESH data none of its keys match.  The old code took
    # `durations.json exists` to mean `durations resolved` and never fell back, so on a
    # skeleton-only fresh DATA_DIR it produced "durations available for 0 clips" while
    # the probe reported duration=True via 'skeleton_json_count' -- T1 (our largest
    # single mechanism, +12 items) silently dead behind a log line claiming it was live.
    # Measured 2026-09-09.  Resolve PER UNIT instead, trying each source in turn.
    durations = {}
    try:
        shipped = {}
        di = HERE / "checkpoints" / "durations.json"
        if di.is_file():
            shipped = json.loads(di.read_text())
            # index by path suffix too: 'Testing/a/b' and 'a/b' must not silently miss
            for k, v in list(shipped.items()):
                parts = [p for p in str(k).split("/") if p]
                for n in (3, 2):
                    if len(parts) >= n:
                        shipped.setdefault("/".join(parts[-n:]), v)

        via = caps.get("duration_via")
        units, n_shipped, n_probe, n_mp4 = [], 0, 0, 0
        seen = set()
        for r in rows:
            vk = common.video_key(str(r.get("path") or ""))
            if vk and vk not in seen:
                seen.add(vk)
                units.append(vk)

        for vk in units:
            n = shipped.get(vk)
            if n:
                durations[vk] = n
                n_shipped += 1
                continue
            if via:                       # the source the probe ADVERTISED -- now used
                try:
                    import modality_probe
                    n = modality_probe.unit_duration(str(data_dir), vk, via)
                except Exception:
                    n = None
                if n:
                    durations[vk] = n
                    n_probe += 1
                    continue
            try:                          # last resort: pure-stdlib mp4 atom walk
                import mp4_frames
                for mp4 in (data_dir / vk).rglob("*.mp4"):
                    n = mp4_frames.frame_count(str(mp4))
                    if n:
                        durations[vk] = n
                        n_mp4 += 1
                    break
            except Exception:
                pass
        log(f"  durations available for {len(durations)} of {len(units)} clips "
            f"(shipped {n_shipped}, probe/{via} {n_probe}, mp4 {n_mp4})")
    except Exception as e:
        log(f"  duration read failed ({type(e).__name__}: {e}); T0 emotion only")

    # ---- BASE: our own model answers EVERY row, replacing the literal 'A' default ----
    # One presence latent per clip (structural evidence always, skeleton when the live
    # DATA_DIR carries it), read out into all six question types. Runs BEFORE the four
    # structural decoders so they still overwrite the rows they fire on.
    try:
        import base_infer
        base = base_infer.predict(rows, str(data_dir), str(HERE / "checkpoints"),
                                  durations=durations, video_key=common.video_key)
        n = sum(1 for k, v in base.items() if pred.get(k) != v)
        pred.update(base)
        log(f"  BASE: {n} rows changed (of {len(rows)})")
    except Exception:
        log("  BASE: FAILED, skipped (falls back to the structural-only pipeline)")
        traceback.print_exc()

    # ---- EFFECTIVE TIER REPORT ----------------------------------------------------
    # Derived from what the run OBSERVABLY DID, never from the probe.  The probe
    # describes the DATA_DIR; it was wrong in both directions (it announced T3_imu /
    # T4_radar, which have no consumer, and reported duration=False on runs where the
    # shipped index then drove T1).  A tier line is a capability claim the organizer's
    # verifier reads, so it is now a statement of observed behaviour.
    try:
        import base_infer as _bi
        n_skel = getattr(_bi, "LAST_SKELETON_CLIPS", 0)
    except Exception:
        n_skel = 0
    tiers = ["T0_structural"]
    if durations:
        tiers.append(f"T1_duration_emotion({len(durations)} clips)")
    if n_skel:
        tiers.append(f"T2_skeleton({n_skel} clips)")
    log(f"tier plan (effective): {tiers}")

    stages = []
    if tbl and dtab and durations:
        stages.append(("emotion_triplet+duration",
                       lambda: emotion_triplet.decode_with_duration(rows, tbl, durations, dtab)))
    elif tbl:
        stages.append(("emotion_triplet", lambda: emotion_triplet.decode(rows, tbl)))
    stages += [
        ("action_decoder",      lambda: action_decoder.decode(rows)),
        ("multi_decoder",       lambda: multi_decoder.decode(rows)),
        ("abstention_recovery", lambda: abstention_recovery.decode(rows)),
    ]
    for name, fn in stages:
        try:
            upd = fn() or {}
            n = sum(1 for k, v in upd.items() if pred.get(k) != v)
            pred.update(upd)
            log(f"  {name}: {n} rows changed")
        except Exception:
            log(f"  {name}: FAILED, skipped (submission still valid)")
            traceback.print_exc()

    common.write_predictions(a.out, rows, pred)
    log(f"wrote {a.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
