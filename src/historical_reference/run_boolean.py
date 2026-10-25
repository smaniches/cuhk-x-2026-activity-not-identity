from pathlib import Path
import csv
import hashlib
import json
from constraint_today import C, possibilities
from rank_constraint_today import scores, choose

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
rows = C.read_qa(ROOT / 'recovered_generators/execution/test_qa.csv')
base = C.read_qa(ROOT / 'recovered_generators/execution/champion_56111316.csv')
pred = {r['qa_id']: r['prediction'] for r in base}
by = C.group_by_video(rows)
ranking = scores(rows)
edits = []
for group in C.segment_siblings(rows):
    members = [r for video in group for r in by[video]]
    allowed = possibilities(members, members)
    if allowed is None:
        raise RuntimeError('Infeasible sibling group: no Boolean solution')
    for q, answers in allowed.items():
        if pred[q] not in answers:
            proposed = choose(ranking[q], answers)
            edits.append(dict(qa_id=q, champion=pred[q], proposed_answer=proposed,
                              possible_answers=answers))
            pred[q] = proposed
output = HERE / 'generated_candidate_A_boolean_constraints.csv'
with output.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['qa_id', 'prediction'])
    writer.writerows((r['qa_id'], pred[r['qa_id']]) for r in rows)
(HERE / 'generated_boolean_edits.json').write_text(json.dumps(edits, indent=2) + '\n')
# Historical predictions are opened only after the generated output is written.
target = C.read_qa(Path('/PARTICIPANT_WORKSPACE/cuhk_large_takeover/candidate_A_boolean_constraints.csv'))
match = sum(pred.get(r['qa_id']) == r['prediction'] for r in target)
sha = hashlib.sha256(output.read_bytes()).hexdigest()
exact = sha == '08bbf447ec60a71c941eaada39fee918ae70d106b588907b9eb714b850353bb5'
print(f'BOOLEAN_MATCH={match}/682\nBOOLEAN_SHA256={sha}\nBOOLEAN_EXACT={"YES" if exact else "NO"}', flush=True)
if not exact:
    raise SystemExit(1)