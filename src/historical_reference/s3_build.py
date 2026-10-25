import pandas as pd, collections, numpy as np
D='/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution/'
tr=pd.read_csv(D+'training_qa.csv'); te=pd.read_csv(D+'test_qa.csv')
sub=pd.read_csv(D+'cand_consistency_v3.csv')
ch=dict(zip(sub.qa_id,sub.prediction))
def n(s): return str(s).strip()
he=tr[(tr.source=='HAU')&(tr.category=='emotion')].copy()
he['s3']=he.path.str.split('/').str[2].str.split('-').str[2].astype(int)
app={s:collections.Counter() for s in [1,2,3]};cor={s:collections.Counter() for s in [1,2,3]}
appA=collections.Counter();corA=collections.Counter()
for _,r in he.iterrows():
    for L in 'ABCD': app[r.s3][n(r[L])]+=1; appA[n(r[L])]+=1
    cor[r.s3][n(r[r.answer])]+=1; corA[n(r[r.answer])]+=1
def rate(tx,s3):
    a=app[s3][tx];c=cor[s3][tx];g=(corA[tx]+1.0)/(appA[tx]+4.0);K=6.0
    return (c+K*g)/(a+K)
e=te[(te.category=='emotion')&(te.source=='HAU')].copy()
e['pid']=e.path.str.extract(r'LM_test_(\d+)').astype(int)
e=e[(e.pid>=65)&(e.pid<=172)]
new=dict(ch); changed=[]
for _,x in e.iterrows():
    s3=((x.pid-65)%3)+1
    o={L:n(x[L]) for L in 'ABCD'}; rt={L:rate(o[L],s3) for L in 'ABCD'}
    top=max(rt,key=rt.get); cL=ch[x.qa_id]
    if top!=cL and rt[cL]<0.10 and rt[top]>=0.60:
        new[x.qa_id]=top; changed.append((x.qa_id,cL,top))
out=sub.copy(); out['prediction']=out.qa_id.map(new)
P=D+'cand_emotion_s3_v1.csv'
out.to_csv(P,index=False)
d=(out.prediction.astype(str)!=sub.prediction.astype(str)).sum()
print(f'wrote {P}')
print(f'rows differing from champion: {d}  (expected 16)')
print('sanity: shape',out.shape,'nulls',out.prediction.isna().sum())
print('changes:',changed)
# verify all other rows identical & format matches sample
ss=pd.read_csv(D+'sample_submission.csv')
print('columns match sample:', list(out.columns)==list(ss.columns), '| qa_id order identical:', (out.qa_id.tolist()==ss.qa_id.tolist()))
