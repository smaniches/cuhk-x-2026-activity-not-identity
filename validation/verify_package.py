#!/usr/bin/env python3
from pathlib import Path
import hashlib, sys, csv
ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
 'submission/final_submission_1.csv':'8e40fdfa92d81f37a96ff70911fb0a2c2a05f9245cccf76bfe579672d1e8fcec',
 'submission/final_submission_2.csv':'9ecbc75529bf89a293722bbc6a44288f07e38c9b446f43c5d8c40a95df7705fc',
 'artifacts/historical_selected/incumbent_079239.csv':'eca1bdfe3e9d757d3a74d4012343a481f9696eaf2b7e71107eb665b91ec58f13',
 'validation/exact_reconstruction/generated_candidate_C_relative_pace.csv':'af72bf032525337fe22cc724d00e5750f106465ce87a903594cfa9d134344a3b',
 'validation/exact_reconstruction/generated_candidate_D_precedence.csv':'a8f318ec2ad4fb93bc3de2516e59fbeef4e6752776190ff9ab63e49a5e5fe404',
 'validation/exact_reconstruction/generated_candidate_F_C2_relative.csv':'8e2dd15accc3039ae7528124450ffb3ef52836bb5e61f1c93995098c8641a304',
 'validation/exact_reconstruction/generated_candidate_L1_seq_margin5.csv':'ced6359a3b23f1f5de3e6a3b8e13d32d8ef85b46772f04ef1b7a0f4801bcf85f',
 'validation/exact_reconstruction/generated_primary_A.csv':'8e40fdfa92d81f37a96ff70911fb0a2c2a05f9245cccf76bfe579672d1e8fcec',
}
failed=False
for rel,exp in EXPECTED.items():
    p=ROOT/rel
    if not p.is_file(): print('MISSING',rel); failed=True; continue
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    print(('OK' if got==exp else 'BAD'),rel,got)
    failed |= got!=exp
# Exact generated primary must be byte-identical to Selected #1.
a=(ROOT/'submission/final_submission_1.csv').read_bytes()
b=(ROOT/'validation/exact_reconstruction/generated_primary_A.csv').read_bytes()
print('PRIMARY_RECONSTRUCTION_BYTE_IDENTICAL', 'YES' if a==b else 'NO')
failed |= a!=b
# No raw challenge QA files should be packaged.
raw=list(ROOT.rglob('training_qa.csv'))+list(ROOT.rglob('test_qa.csv'))
print('RAW_CHALLENGE_QA_FILES',len(raw))
failed |= bool(raw)
sys.exit(1 if failed else 0)
