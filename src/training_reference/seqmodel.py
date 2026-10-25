import numpy as np, pandas as pd, itertools, re, sys, json, collections, time, os
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,'/PARTICIPANT_WORKSPACE/historical_seq_workspace/cuhk_large_seq')
from action_map import HARN_TO_CANON
WF=os.environ.get('WF','win_W2.0_S1.0.npz'); C_LR=float(os.environ.get('C',0.3))
d=np.load(WF,allow_pickle=True); units=list(d['units']); X={u:x for u,x in zip(units,d['X'])}; Cn={u:c for u,c in zip(units,d['C'])}
tr=pd.read_csv('/PARTICIPANT_WORKSPACE/cuhk_work/large/training_qa.csv'); te=pd.read_csv('/PARTICIPANT_WORKSPACE/cuhk_work/large/test_qa.csv')
classes=sorted(set(HARN_TO_CANON.values())); ci={c:i for i,c in enumerate(classes)}
def user_of(p):
    m=re.search(r'user(\d+)',p); return f'user{m.group(1)}' if m else 'test'
# HARn windows
hX=[];hy=[];hg=[]
for u in units:
    if '/HARn/' not in u: continue
    act=HARN_TO_CANON[u.split('/')[2]]
    x=X[u]; hX.append(x); hy+= [ci[act]]*len(x); hg+=[user_of(u)]*len(x)
hX=np.concatenate(hX); hy=np.array(hy); hg=np.array(hg)
print('HARn windows',hX.shape,'users',len(set(hg)),flush=True)
users=sorted(set(hg)); gkf=GroupKFold(6); fold_of={}
folds=list(gkf.split(hX,hy,groups=hg))
for k,(a,b) in enumerate(folds):
    for u in set(hg[b]): fold_of[u]=k
def fit(mask):
    sc=StandardScaler().fit(hX[mask]); m=LogisticRegression(max_iter=2000,C=C_LR).fit(sc.transform(hX[mask]),hy[mask]); return sc,m
t0=time.time(); models={}
for k,(a,b) in enumerate(folds):
    msk=np.zeros(len(hy),bool); msk[a]=True; models[k]=fit(msk); print('fold',k,f'{time.time()-t0:.0f}s',flush=True)
models['full']=fit(np.ones(len(hy),bool))
def post(u,key):
    sc,m=models[key]; p=m.predict_proba(sc.transform(X[u])); out=np.zeros((len(p),len(classes)),np.float32); out[:,m.classes_]=p; return out
# sequence rows
def rows(df,is_test):
    s=df[df.category=='sequence'].copy()
    s['unit']=('Testing/'+s.path.str.split('/').str[:2].str.join('/')) if is_test else ('Training/'+s.path)
    s['user']=s.path.map(user_of); return s
S=rows(tr,False); St=rows(te,True)
print('train seq',len(S),'units present',S.unit.isin(X).sum(),'test',len(St),St.unit.isin(X).sum(),flush=True)
def sensor_pair(P,opts):
    # P (T,40); returns dict (a,b)->logit a before b
    T=len(P); out={}
    for a,b in itertools.permutations(opts,2):
        if (b,a) in out: out[(a,b)]=-out[(b,a)]; continue
        pa=P[:,ci[a]]; pb=P[:,ci[b]]
        cb=np.cumsum(pb); tot=cb[-1]
        before=np.sum(pa*(tot-cb))  # a at t, b after t
        after=np.sum(pa*(cb-pb))     # b before t
        out[(a,b)]=np.log((before+1e-6)/(after+1e-6))
    return out
# prior pairwise counts per user (for LOSO exclusion)
pc=collections.defaultdict(collections.Counter)
for _,r in S.iterrows():
    o=[r[c] for c in r.answer]
    for i in range(4):
        for j in range(i+1,4): pc[r.user][(o[i],o[j])]+=1
def prior_logit(excl):
    tot=collections.Counter()
    for u in pc:
        if u!=excl: tot.update(pc[u])
    return lambda a,b: np.log((tot[(a,b)]+0.5)/(tot[(b,a)]+0.5))
PERMS=list(itertools.permutations(range(4)))
def decode(opts,sp,pl,ws,wp):
    best=None;scores=[]
    for perm in PERMS:
        o=[opts[i] for i in perm]; s=0
        for i in range(4):
            for j in range(i+1,4): s+=ws*sp[(o[i],o[j])]+wp*pl(o[i],o[j])
        scores.append(s)
    scores=np.array(scores); k=scores.argmax(); srt=np.sort(scores)
    return ''.join('ABCD'[i] for i in PERMS[k]), srt[-1]-srt[-2]
SP={}
for _,r in S.iterrows():
    SP[r.qa_id]=sensor_pair(post(r.unit,fold_of.get(r.user,'full')),[r.A,r.B,r.C,r.D])
res={}
for ws in [0,0.25,0.5,1,2,4,1000]:
    acc=0; pw=0
    for _,r in S.iterrows():
        pl=prior_logit(r.user); ans,_=decode([r.A,r.B,r.C,r.D],SP[r.qa_id],pl,ws,1 if ws<1000 else 0)
        acc+=ans==r.answer
    res[ws]=acc/len(S); print(f'ws={ws} LOSO exact={acc/len(S):.4f}',flush=True)
json.dump({str(k):v for k,v in res.items()},open(f'seq_res_{os.path.basename(WF)}.json','w'))
# test decode with best ws
best_ws=max([k for k in res if k<1000],key=lambda k:res[k])
pl=prior_logit(None); out=[]
for _,r in St.iterrows():
    sp=sensor_pair(post(r.unit,'full'),[r.A,r.B,r.C,r.D])
    ans,m=decode([r.A,r.B,r.C,r.D],sp,pl,best_ws,1); ans0,m0=decode([r.A,r.B,r.C,r.D],sp,pl,0,1)
    out.append((r.qa_id,ans,m,ans0))
pd.DataFrame(out,columns=['qa_id','pred','margin','prior_only']).to_csv(f'seq_test_{os.path.basename(WF)}.csv',index=False)
print('best ws',best_ws,'saved')
