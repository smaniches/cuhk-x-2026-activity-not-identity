# Activity, Not Identity

**CUHK-X Challenge 2026 · Large Model Track · Santiago Maniches**

Session-level structural decoding for non-RGB human-activity question answering. This repository publishes the author's code, inference checkpoint, and documented historical prediction artifacts for the CUHK-X Challenge 2026. It is the public distribution of work originally submitted to the organizers in September 2026, not a new competition entry.

**Read this distinction before quoting results:** The selected Kaggle submission and the independently executable CPU inference runtime are **two different artifacts**. The selected prediction file obtained a **0.90882 private Kaggle score** (primary submission ID `56241648`). The CPU runtime is a different, independently fitted reference implementation and **does not reproduce those selected predictions byte for byte on raw inputs**. Its subject-held-out performance is reported in the submitted technical report, not newly established by this public distribution. No raw-to-selected-score end-to-end reproduction is claimed.

## What runs

The provided `inference.sh` executes the **fresh-data CPU reference runtime** with the shipped linear-model checkpoint and deterministic session decoders. It reads an organizer-format question CSV and optionally matching skeleton pose files and clip-duration evidence. No Kaggle prediction CSV, hidden labels, external API, cloud model, or LLM is read during this inference path.

**Requirements:** Linux or compatible system, Python 3.11+ (the archived verification environment specifies Python 3.13.5), NumPy `2.3.5`, Bash. No GPU and no network access during inference. Network access may be required once to install NumPy.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
bash inference.sh /path/to/official-cuhk-input /path/to/predictions.csv
```

The input directory must contain `test_qa.csv` (also accepts `test.csv` or `qa/test_qa.csv`), with the official `qa_id, source, path, category, question, A, B, C, D` fields. For sensor-backed inference, place the organizer-provided per-clip `Skeleton/predictions/*.json` at the clip paths referenced by the question table. Where available, the runtime can use frame counts from skeleton files or Depth MP4 headers; it **does not decode image pixels**. The output is a CSV with `qa_id,prediction` in input order. For HAU, the session decoder uses question relationships that reflect the benchmark's recording and annotation protocol. Ordinary MP4 videos or arbitrary scene-description JSON files are *not* valid direct inputs.

The public distribution omits a path-keyed duration cache extracted from the CUHK-X dataset in order not to redistribute source-derived per-clip metadata. The unchanged source code can calculate durations from *authorized local data* supplied at runtime. This omission means QA-only outputs without media may differ from historical cached-duration runs; it does not alter the trained model weights or decoding logic.

Test the command and failure behavior locally, without access to the original challenge data:

```bash
python tests/smoke_inference.py
python validation/verify_package.py
```

The smoke test generates **synthetic** question rows and skeleton coordinates and only checks execution, output format, and determinism. It does not measure scientific accuracy. The package verifier checks hashes of **stored** historical prediction artifacts; it does *not* regenerate the original selected predictions.

## What was scored versus what is preserved

| Item | Evidence and runnable status |
| --- | --- |
| Primary Kaggle selection (`56241648`) | Preserved byte-identical CSV in `submission/final_submission_1.csv`, SHA-256 `8e40fdfa92d81f37a96ff70911fb0a2c2a05f9245cccf76bfe579672d1e8fcec`; published Kaggle private score `0.90882` (reported). |
| Secondary selection (`56254266`) | Preserved CSV under `submission/final_submission_2.csv`; no recovered executable secondary pipeline. |
| Historical stage records | Preserved source in `src/historical_reference/` and stage-output CSVs in `validation/exact_reconstruction/`. The source uses unavailable historical workspaces and **is for static review**, not an executable end-to-end reconstruction from this repository. |
| Independent fresh-data implementation | `inference.sh` and `src/fresh_runtime/`; trained, compact NumPy checkpoint plus deterministic structural decoders; runnable on authorized CUHK-format data. |
| Historical training reference | `src/training_reference/`; archived feature/fit and sequence scripts, with additional scientific Python dependencies and inputs not bundled here. This is not a push-button retraining package. |

The selected submission's lineage started from a **publicly released prediction vector from team Fususu** (Kaggle author `phuongncn`, public baseline score `0.77777`), then applied the author's documented repairs and structural decoding stages. **The original generator of that upstream vector was not released by its authors**, and other historical intermediate inputs are not all present. The executable CPU implementation instead uses the author's own fitted base classifier and does not load that vector. See [Provenance](docs/PROVENANCE.md), the submitted technical report and [Third-party rights](docs/THIRD_PARTY.md). No third-party code is claimed as authored here.

The historical archive was submitted to the organizing committee on September 18, 2026, with the limitations disclosed in its README and the September 30 report. The original archive SHA-256 is `6cc1f271800115b04d76821e7a07d7eb2eb00af4ac86d257185ee689f21df86c` (4,884,902 bytes). This GitHub distribution is intentionally **not byte-identical to the original archive**: signed declarations and dataset-derived/nonportable supplementary files are excluded. [Original organizer document snapshots](docs/historical_record/) are retained solely to document the original verification scope, not to redefine the contents of this public repository.

## Scientific scope

The reported benefit of session-level decoding relies on properties of the CUHK-X question-construction protocol: shared session context, recurring true-action options, and speed-dependent emotion options. These are experimentally described in the submitted report, **not demonstrated as general-purpose video understanding**. They have not been shown to generalize to independently written video questions.

**Technical report:** *Activity, Not Identity: Session-level structural decoding for privacy-preserving human-activity question answering* (Santiago Maniches, submitted September 30, 2026). The technical-report PDF for the CUHK community deposit is prepared separately; it contains the original report plus a transparent publication-only URL addendum. The Zenodo DOI will be added only after community approval.

## Licensing and attribution

**Copyright © 2026 Santiago Maniches.** The author's original code is provided under [Apache License 2.0](LICENSE), as requested by the CUHK-X Challenge organizers. Authored documentation in this repository may be reused under the same license unless otherwise indicated. This does **not** grant ownership of or relicense CUHK-X data, third-party code, prior public predictions, or other protected source material. Read [NOTICE](NOTICE) and [Third-party rights](docs/THIRD_PARTY.md) before redistributing any artifacts.

The CUHK-X dataset and source annotations are owned by **CUHK AIoT Lab** and subject to the dataset's licensing and access rules; download them from the organizers under the applicable agreement. No raw question table, label data, skeleton frames, media or signed honor declaration is distributed here.

**Dataset citation:** S. Jiang et al., *A Large-Scale Multimodal Dataset and Benchmarks for Human Activity Scene Understanding and Reasoning*, MobiSys 2026, DOI: [10.1145/3745756.3809209](https://doi.org/10.1145/3745756.3809209). The public starting-vector source is [Fususu's Kaggle research handoff](https://www.kaggle.com/code/phuongncn/lb-0-77777-exact-submission-research-handoff).