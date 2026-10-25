import pandas as pd, numpy as np, os
from base import load
D='/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution'
tr,te,hau=load()
champ=pd.read_csv(os.path.join(D,'cand_emotion_s3_v1.csv'))
C=pd.read_csv('ranked_candidates.csv')
edits=C[(C.region=='clean')&(C.changed)][['qa_id','new_letter']]
print('edits to apply: %d'%len(edits))
out=champ.copy()
mp=dict(zip(edits.qa_id,edits.new_letter))
out['prediction']=[mp.get(q,p) for q,p in zip(out.qa_id,out.prediction)]

# ---------------- VALIDATION ----------------
base=pd.read_csv(os.path.join(D,'test_qa.csv'))
base.columns=[c.strip().lstrip('﻿') for c in base.columns]
assert list(out.qa_id)==list(base.qa_id), 'qa_id order must match test_qa.csv'
assert len(out)==682, len(out)
assert out.prediction.notna().all() and (out.prediction.astype(str).str.len()>0).all()
bad=[p for p in out.prediction if not set(str(p))<=set('ABCD')]
assert not bad, bad[:5]
m=out.merge(base[['qa_id','category']],on='qa_id')
assert m[m.category=='multi'].prediction.astype(str).str.len().min()>=1
diff=(out.prediction.values!=champ.prediction.values).sum()
print('rows: %d   order matches test_qa.csv: True   all letters in A-D: True'%len(out))
print('multi rows non-empty: %d/%d'%((m[m.category=='multi'].prediction.astype(str).str.len()>0).sum(),(m.category=='multi').sum()))
print('differences vs champion cand_emotion_s3_v1.csv: %d (all emotion): %s'%(
    diff, set(m[out.prediction.values!=champ.prediction.values].category)))
P='/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution/cand_emotion_core_v2.csv'
out.to_csv(P,index=False)
print('WROTE', P)
print(pd.read_csv(P).head(3).to_string())

# ---------------- ranked table for the report ----------------
R=C[(C.region=='clean')&(C.changed)].copy()
R['exp_delta']=R.P_new-R.P_champ
R=R.sort_values('exp_delta',ascending=False)
R[['pid','s3','post_new','champ','new']] = R[['pid','s3','P_new','champ_word','new_word']]
cols=['pid','s3','core','P_new','champ_letter','champ_word','P_champ','new_letter','new_word','exp_delta']
R[cols].to_csv('ranked_edit_table.csv',index=False)
print()
print('=== RANKED EDIT SET (%d rows), by expected item delta ==='%len(R))
print(R[cols].round(3).to_string(index=False))
