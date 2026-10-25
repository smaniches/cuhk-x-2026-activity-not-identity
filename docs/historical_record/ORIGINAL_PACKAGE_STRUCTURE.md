# Package structure

This folder follows the CUHK-X verification-package template.

- `README.md` - organizer run guide and known limitation disclosure.
- `VERIFICATION_SCOPE.md` - precise distinction between exact historical reconstruction evidence and the fresh-data executable runtime.
- `manifest.yaml` - package/environment/artifact manifest.
- `inference.sh` - single organizer entry point for the fresh-data runtime.
- `configs/` - frozen package configuration.
- `environment/` - pinned Python requirements and setup notes.
- `src/fresh_runtime/` - label-free arbitrary-data inference path called by `inference.sh`.
- `src/historical_reference/` - historical Selected-submission source preserved for static review only.
- `src/training_reference/` - fitting/training code requested by the organizer; no competition data included.
- `artifacts/historical_selected/` - historical frozen prediction/checkpoint artifacts disclosed for validation only.
- `submission/` - exact byte copies of both Kaggle Selected CSVs plus mapping metadata.
- `validation/exact_reconstruction/` - exact Selected-A reconstruction hashes and generated stage artifacts; not reachable from `inference.sh`.
- `declarations/` - completed signed Honor Declaration PDF.
- `SHA256SUMS.txt` - SHA-256 for every packaged file.
