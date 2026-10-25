from pathlib import Path
import hashlib
import json
from joint_extension_today import C, run

HERE = Path(__file__).resolve().parent
rows = C.read_qa(HERE.parent / 'recovered_generators/execution/test_qa.csv')
base_rows = C.read_qa(HERE / 'generated_candidate_A_boolean_constraints.csv')
pred = {r['qa_id']: r['prediction'] for r in base_rows}
new, groups = run(rows, pred)
protected = {e["qa_id"] for e in json.loads((HERE / "generated_boolean_edits.json").read_text())}
new = {q: a for q, a in new.items() if q not in protected}
edits = [dict(qa_id=q, previous=pred[q], prediction=a, group=groups[q])
         for q, a in new.items() if a != pred[q]]
pred.update(new)
output = HERE / 'generated_candidate_B_joint_siblings.csv'
C.write_predictions(output, rows, pred)
(HERE / 'generated_joint_edits.json').write_text(json.dumps(edits, indent=2) + '\n')
# Validate only after generation.
target_path = Path('/PARTICIPANT_WORKSPACE/cuhk_large_takeover/frozen/champion_56148570.csv')
target = C.read_qa(target_path)
differences = [dict(qa_id=r['qa_id'], generated=pred.get(r['qa_id']), expected=r['prediction'])
               for r in target if pred.get(r['qa_id']) != r['prediction']]
sha = hashlib.sha256(output.read_bytes()).hexdigest()
exact = sha == hashlib.sha256(target_path.read_bytes()).hexdigest()
print(f'JOINT_MATCH={len(target)-len(differences)}/682\nJOINT_SHA256={sha}\nJOINT_EXACT={"YES" if exact else "NO"}', flush=True)
if not exact:
    print(json.dumps(differences, indent=2))
    raise SystemExit(1)