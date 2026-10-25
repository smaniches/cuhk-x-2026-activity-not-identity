# Team Santiago Maniches Reproduction Guide

## 0 Needs attention

- More than one GPU needed: `no`
- More than 24 GB GPU memory needed: `no`
- Specific GPU generation or driver needed: `no`
- Network access needed at inference time: `no`
- Runs on Linux x86_64: `yes`

Execution scope: both exact Selected Kaggle CSVs are included byte-for-byte. The historical Selected-A reconstruction included under `validation/exact_reconstruction/` is hash-identical to the primary Selected CSV from its preserved frozen historical state. The top-level `inference.sh` is a separate label-free fresh-data reference runtime and is not represented as byte-identical to the primary Selected submission. This distinction is intentional and explicit; no original-test answer-lookup branch is used.

## 1 Team and submission

- Team name: `Santiago Maniches`
- Track: `large`
- Contact: `santiago.maniches@gmail.com`
- Submission record: `submission/submission_info.yaml`
- Primary Selected submission: ID `56241648`, file `CUHK_A_sha8e40fdfa92d8.csv`, UTC `2026-09-15T00:33:51.303Z`, public `0.94444`, private `0.90882`, SHA-256 `8e40fdfa92d81f37a96ff70911fb0a2c2a05f9245cccf76bfe579672d1e8fcec`
- Second Selected submission: ID `56254266`, file `CUHK_C_independent_union.csv`, UTC `2026-09-15T12:29:51.067Z`, public `0.94444`, private `0.90294`, SHA-256 `9ecbc75529bf89a293722bbc6a44288f07e38c9b446f43c5d8c40a95df7705fc`
- Primary verification target: `56241648`
- Solution type: `hybrid`

The two Selected CSVs are preserved byte-for-byte and were not resaved or reordered.

## 2 Frozen solution summary

The package contains:

1. a deterministic CPU-only, label-free fresh-data runtime under `src/fresh_runtime/`; and
2. preserved historical artifacts and validation evidence for the exact Selected-A reconstruction.

The fresh runtime uses a frozen NumPy checkpoint plus deterministic structural decoding. It does not read Kaggle submission CSVs, historical prediction CSVs, Training labels, hidden labels, external APIs, or network services at inference time.

The exact Selected-A reconstruction evidence is kept separate from the top-level inference path. Its earliest preserved historical prediction state is included under `artifacts/historical_selected/`. The raw-data executable that originally generated that preserved state is not included in the recovered materials represented by this package.

The second Selected CSV is preserved for static review. A distinct end-to-end executable secondary configuration is not represented as recovered.

## 3 Hardware and runtime

- OS: Linux x86_64
- GPU: none required
- GPU count: 0
- Minimum GPU memory: 0 GB
- Minimum host RAM: 4 GB
- Free disk: 1 GB
- Python tested: 3.13.5
- NumPy tested: 2.3.5
- Setup time: approximately 2 minutes when Python and pip are available
- Inference: CPU; runtime depends on supplied media volume

## 4 Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r environment/requirements.txt
```

No network access is required after setup. No GPU, CUDA, or external inference service is required.

## 5 Run inference

```bash
bash inference.sh <test_data_dir> <output_csv>
```

Equivalent explicit primary ID:

```bash
bash inference.sh <test_data_dir> <output_csv> 56241648
```

The script locates `test_qa.csv` below `<test_data_dir>` (also accepting `test.csv` or `qa/test_qa.csv`) and writes `qa_id,prediction` in input row order.

The top-level runtime is the fresh-data reference runtime described above; it is not asserted to be byte-identical to the primary Selected submission. Passing secondary ID `56254266` exits non-zero rather than silently substituting a different configuration.

## 6 Artifacts and external resources

Fresh runtime artifacts:

- `src/fresh_runtime/checkpoints/base_v1.npz`
- `src/fresh_runtime/checkpoints/base_v1.json`
- `src/fresh_runtime/checkpoints/emotion_slot_table.json`
- `src/fresh_runtime/checkpoints/emotion_duration_table.json`
- `src/fresh_runtime/checkpoints/durations.json`

Historical validation artifacts:

- `artifacts/historical_selected/incumbent_079239.csv`
- `artifacts/historical_selected/post_win_W2.0_S1.0.npz`
- `artifacts/historical_selected/post_win_W3.0_S1.0.npz`

No external training dataset, pretrained-model download, closed-source API, or external inference service is required by the top-level runtime. Competition data are not included.

## 7 Reproduction expectations

The top-level runtime is deterministic for identical inputs and environment. No inference-time random seed is used.

The exact historical Selected-A reconstruction evidence is byte-identical to the primary Selected CSV and matches its SHA-256 exactly. It is provided as validation evidence and is not used as a hidden answer lookup by `inference.sh`.

## 8 Known limitations and troubleshooting

- The top-level fresh-data runtime is not represented as byte-identical to Selected submission `56241648`.
- The raw-data executable for the earliest preserved historical prediction state is not included in the recovered materials represented by this package.
- A separate executable path for Selected submission `56254266` is not represented as recovered.
- No replacement model, prompt, or original-test-specific answer branch is included.

## 9 Final checklist

- [x] Required template fields completed.
- [x] `manifest.yaml` matches packaged artifacts.
- [x] Top-level runtime uses relative package paths and organizer-supplied input path.
- [x] Both exact Selected CSVs and hashes are included.
- [x] Primary verification ID is `56241648`.
- [x] Runtime network/API requirements are declared.
- [x] Official Honor Declaration reviewed and signed; `declarations/honor_declaration.pdf` included.
- [x] No API keys, passwords, raw challenge dataset, internal transcripts, or participant virtual environment are included.
