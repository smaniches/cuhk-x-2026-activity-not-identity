# Provenance and evidence boundary

The immutable organizer-verification archive was submitted September 18, 2026:

- Name: `large_Santiago_Maniches_verification_v1.zip`
- Size: `4,884,902` bytes
- SHA-256: `6cc1f271800115b04d76821e7a07d7eb2eb00af4ac86d257185ee689f21df86c`
- Team: Santiago Maniches; Large Model Track
- Primary selected Kaggle ID: `56241648`; private leaderboard `0.90882`
- Secondary ID: `56254266`; private leaderboard `0.90294`

The original archive's internal `SHA256SUMS.txt` names 70 files, including files intentionally omitted from this public subset. The original checksum manifest is preserved in `docs/historical_record/ORIGINAL_PACKAGE_SHA256SUMS.txt` for provenance; it is NOT the active checksum list for this repository.

## Claims separated

1. **Selected CSV preservation (verified)**. `submission/final_submission_1.csv` has SHA-256 `8e40fdfa92d81f37a96ff70911fb0a2c2a05f9245cccf76bfe579672d1e8fcec`; `validation/exact_reconstruction/generated_primary_A.csv` has identical bytes. `validation/verify_package.py` verifies stored files.
2. **Historical staged reconstruction (archived evidence, not rerunnable here)**. Stage files and historical source are preserved. Some scripts require unshipped original intermediate files and workspace paths. Neither the published repository nor the original verification ZIP offers a self-contained raw-data-to-Kaggle-selected-file generator.
3. **Fresh-data runtime (runnable implementation, not Kaggle-selected pipeline)**. `inference.sh` executes a NumPy-based reference model with the author's fitted weights, without loading the historical predictions. It requires authorized organizer-format question/sensor inputs to reproduce the quantitative measurements in the original report.
4. **Reported measurements, not remeasured by this release**. Primary private leaderboard 0.90882; held-out training-subject validation 0.900; historical agreement of the independent runtime with 598/682 selected-file answers when skeleton data are supplied. These are documented in the September 30 submitted technical report; a synthetic smoke test establishes no accuracy statistic.

The original PDF is dated September 30, 2026. Public-release URL metadata added later is administrative and does not alter the historical report's science, scores, or experiment claims.

## Public release omissions

- `declarations/` (personally signed honor declaration; private organizer material).
- `src/fresh_runtime/checkpoints/durations.json` (per-clip cache of organizer-data-derived frame counts; avoid redistributing such metadata under the CUHK-X dataset terms). The unchanged runtime can infer durations from authorized supplied files.
- `artifacts/historical_selected/post_win_W*.npz` (historical, nonportable object-serialized intermediary arrays; not used by `inference.sh`). The original archive records their SHA-256 values.
- Original umbrella `manifest.yaml` retained as `docs/historical_record/ORIGINAL_PACKAGE_MANIFEST.yaml`, because its file listing corresponds to the private submission, not the public subset.

All executable inference source and trained base checkpoint files included in this repository are byte-for-byte copied from the September 18 submission. No algorithm was rewritten for publication.
