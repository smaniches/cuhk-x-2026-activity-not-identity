import pandas as pd, numpy as np, collections, itertools, re, os
D='/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution'

def load():
    tr = pd.read_csv(os.path.join(D,'training_qa.csv'))
    te = pd.read_csv(os.path.join(D,'test_qa.csv'))
    tr.columns=[c.strip().lstrip('﻿') for c in tr.columns]
    te.columns=[c.strip().lstrip('﻿') for c in te.columns]
    hau = tr[tr.source=='HAU'].copy()
    ex = hau.path.str.extract(r'HAU/user(\d+)/(\d+)-(\d+)-(\d+)')
    for i,c in enumerate(['user','s1','s2','s3']): hau[c]=ex[i].astype(int)
    te['pid']=te.path.str.extract(r'LM_test_(\d+)')[0].astype(int)
    return tr,te,hau

def anstext(df):
    return [r[r['answer']] for _,r in df.iterrows()]

# ---- ORDERING of train HAU paths, to mimic the test sorted re-index ----
def sortkey_str(p):
    return p   # plain lexicographic on full path string

if __name__=='__main__':
    tr,te,hau=load()
    em=hau[hau.category=='emotion'].copy()
    em['ans']=anstext(em)
    # --- option presence vs s3 (are distractors s3-dependent?) ---
    rows=[]
    for _,r in em.iterrows():
        for c in 'ABCD':
            rows.append((r[c], r.s3, c==r['answer']))
    o=pd.DataFrame(rows,columns=['w','s3','is_ans'])
    pres = pd.crosstab(o.w,o.s3)
    dist = pd.crosstab(o[~o.is_ans].w, o[~o.is_ans].s3)
    n3 = em.s3.value_counts().sort_index()
    print('rows per s3:', dict(n3))
    dd = dist.reindex(pres.index).fillna(0).astype(int)
    dd['tot']=dd.sum(1)
    dd=dd.sort_values('tot',ascending=False)
    # chi2-ish check on distractor distribution
    from scipy.stats import chi2_contingency
    sub=dd[dd.tot>=20][[1,2,3]]
    chi2,p,dof,exp=chi2_contingency(sub.values)
    print('DISTRACTOR-only presence vs s3: chi2=%.1f dof=%d p=%.4g  (words n>=20: %d)'%(chi2,dof,p,len(sub)))
    print(dd.head(25).to_string())
