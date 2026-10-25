#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: bash inference.sh <test_data_dir> <output_csv> [submission_id]" >&2
  exit 64
fi
DATA_DIR="$1"
OUTPUT_CSV="$2"
SUBMISSION_ID="${3:-56241648}"
case "$SUBMISSION_ID" in
  56241648)
    exec python3 "$(dirname "$0")/src/fresh_runtime/run_inference.py" "$DATA_DIR" -o "$OUTPUT_CSV"
    ;;
  56254266)
    echo "ERROR: Selected submission 56254266 is preserved for static mapping review, but an independent executable secondary path was not recovered. See README section 0 and submission/submission_info.yaml." >&2
    exit 3
    ;;
  *)
    echo "ERROR: unknown Selected submission ID: $SUBMISSION_ID" >&2
    exit 64
    ;;
esac
