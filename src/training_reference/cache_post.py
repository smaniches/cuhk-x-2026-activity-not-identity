import numpy as np, pandas as pd, re, sys, os, time
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
sys.path.insert(0,'/PARTICIPANT_WORKSPACE/historical_seq_workspace/cuhk_large_seq')
from action_map import HARN_TO_CANON
WF=sys.argv[1]; C_LR=float(sys.argv[2]) if len(sys.argv)>2 else 0.3
d=np.load(WF,allow_pickle=True); units=list(d['units']); X={u:x for u,x in zip(units,d['X'])}
classes=sorted(set(HARN_TO_CANON.values())); ci={c:i for i,c in enumerate(classes)}
def user_of(p):
    m=re.search(r'user(\d+)',p); return f'user{m.group(1)}' if m else 'test'
hX=[];hy=[];hg=[]
for u in units:
    if '/HARn/' not in u: continue
    act=HARN_TO_CANON[u.split('/')[2]]; x=X[u]; hX.append(x); hy+=[ci[act]]*len(x); hg+=[user_of(u)]*len(x)
hX=np.concatenate(hX); hy=np.array(hy); hg=np.array(hg)
folds=list(GroupKFold(6).split(hX,hy,groups=hg)); fold_of={}
for k,(a,b) in enumerate(folds):
    for u in set(hg[b]): fold_of[u]=k
def fit(idx):
    sc=StandardScaler().fit(hX[idx]); m=LogisticRegression(max_iter=2000,C=C_LR).fit(sc.transform(hX[idx]),hy[idx]); return sc,m
t0=time.time()
ms=Parallel(n_jobs=7)(delayed(fit)(a) for a,b in folds+[(np.arange(len(hy)),None)])
models={k:ms[k] for k in range(6)}; models['full']=ms[6]
print('fit',f'{time.time()-t0:.0f}s',flush=True)
# HARn OOF acc
acc=0
for k,(a,b) in enumerate(folds):
    sc,m=models[k]; acc+=(m.predict(sc.transform(hX[b]))==hy[b]).sum()
print('HARn window OOF acc',acc/len(hy))
out={}
for u in units:
    if '/HARn/' in u: continue
    key=fold_of.get(user_of(u),'full') if u.startswith('Training') else 'full'
    sc,m=models[key]; p=m.predict_proba(sc.transform(X[u])); P=np.zeros((len(p),len(classes)),np.float32); P[:,m.classes_]=p; out[u]=P
np.savez_compressed('post_'+os.path.basename(WF),units=np.array(list(out)),P=np.array([out[u] for u in out],dtype=object),classes=np.array(classes))
print('saved',len(out))
