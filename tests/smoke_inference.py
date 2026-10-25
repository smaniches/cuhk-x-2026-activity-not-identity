#!/usr/bin/env python3
"""Synthetic input/output contract test only; does NOT validate competition accuracy."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COLS = ["qa_id", "source", "path", "category", "question", "A", "B", "C", "D"]
ROWS = [
    ["synthetic_1", "HAU", "synthetic/a/Depth/Depth.mp4", "single", "Which action?", "Walking", "Standing", "Stretching", "Waving"],
    ["synthetic_2", "HAU", "synthetic/a/Depth/Depth.mp4", "emotion", "How?", "Calmly", "Hastily", "Leisurely", "Restlessly"],
    ["synthetic_3", "HAU", "synthetic/b/Depth/Depth.mp4", "emotion", "How?", "Leisurely", "Calmly", "Hastily", "Softly"],
    ["synthetic_4", "HAU", "synthetic/c/Depth/Depth.mp4", "emotion", "How?", "Hastily", "Calmly", "Leisurely", "Diligently"],
    ["synthetic_5", "HARn", "synthetic/d/Depth/Depth.mp4", "single", "What action?", "Waving", "Standing", "Jumping", ""],
]

def run(*args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(["bash", str(ROOT / "inference.sh"), *map(str, args)],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)

with tempfile.TemporaryDirectory() as td:
    temp = Path(td)
    data = temp / "input"
    data.mkdir()
    with (data / "test_qa.csv").open("w", newline="", encoding="utf-8") as stream:
        out = csv.writer(stream)
        out.writerow(COLS)
        out.writerows(ROWS)
    for name in "abcd":
        path = data / "synthetic" / name / "Skeleton" / "predictions"
        path.mkdir(parents=True)
        for t in range(14):
            joints = [[j * 0.08 + t * 0.01, j * 0.05 + t * 0.015, 0.1 + j * 0.01] for j in range(17)]
            (path / f"{t:05}.json").write_text(json.dumps(
                [{"keypoints": joints, "keypoint_scores": [0.85] * 17}]))
    paths = [temp / "first.csv", temp / "second.csv"]
    for target in paths:
        result = run(data, target)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert "skeleton on 4" in result.stdout, result.stdout
        assert "HARn arm on 1" in result.stdout, result.stdout
        with target.open(newline="", encoding="utf-8") as f:
            recs = list(csv.DictReader(f))
        assert len(recs) == len(ROWS)
        assert [r["qa_id"] for r in recs] == [r[0] for r in ROWS]
        assert all(set(r["prediction"]) <= set("ABCD") and r["prediction"] for r in recs)
    assert paths[0].read_bytes() == paths[1].read_bytes(), "not deterministic"
    assert run(data, temp / "n.csv", 56254266).returncode == 3
    assert run(data, temp / "n.csv", 0).returncode == 64
    assert run().returncode == 64
    assert run(temp / "absent", temp / "n.csv").returncode != 0
    print("PASS: synthetic inference format, skeleton path, HARn arm, determinism, invalid-ID gates")
    print("synthetic-output-SHA256", hashlib.sha256(paths[0].read_bytes()).hexdigest())
