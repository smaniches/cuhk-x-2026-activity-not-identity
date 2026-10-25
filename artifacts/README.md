# Packaged artifacts

Runtime checkpoints called by `inference.sh` are stored under `src/fresh_runtime/checkpoints/` and enumerated in `manifest.yaml`.

`artifacts/historical_selected/` contains disclosed historical Selected-submission artifacts used only for reconstruction/provenance review. None of those historical prediction files are reachable from the top-level fresh-data runtime.
