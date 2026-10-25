# Third-party materials and rights

The author's original software implementation is offered under Apache-2.0 to satisfy the finalist code-release request. This license does not override the rights of CUHK or third parties.

## CUHK-X dataset

CUHK AIoT Lab owns the dataset, annotations, media, splits and related source metadata. The current CUHK-X repository license describes dataset access as noncommercial and not redistributable, with separate rules for annotations and metadata. **Do not upload or bundle official training/test question tables, skeleton JSON, depth videos, labels, individual subject data, or per-clip metadata extracted from the dataset.** Obtain input data separately and follow the organizer's applicable terms and authorizations.

Official: https://github.com/openaiotlab/CUHK-X
Paper: https://doi.org/10.1145/3745756.3809209

## Public starting prediction vector

The selected Kaggle lineage used a public baseline prediction vector from team Fususu (Kaggle user `phuongncn`), public score `0.77777`, described and credited in the original technical report. See:
https://www.kaggle.com/code/phuongncn/lb-0-77777-exact-submission-research-handoff

The published handoff documents that its own illustrative clip-graph decoder did not produce its archived baseline vector and that the original sensor-specialist generator/checkpoints were not released. The author's independent CPU inference runtime here does not read this vector. No third-party authorship is claimed.

## Submission outputs

The submitted prediction CSVs contain authored predictions, with row identifiers originating from the competition question schema. They are historical result artifacts, **not a redistribution of raw labels or sensor data**. Users must independently obtain and use official data under the organizer's applicable terms.
