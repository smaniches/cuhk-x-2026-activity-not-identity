"""FULL end-to-end second-generation pipeline, validated on TRAIN with LOO."""
import pandas as pd, numpy as np, collections, itertools, math
from base import load, anstext

THR=0.40
def jac(a,b): return len(a&b)/len(a|b)

def segment_runs(O, thr=THR):
    """link adjacent option-sets with jaccard>=thr -> runs"""
    runs=[]; cur=[0]
    for i in range(len(O)-1):
        if jac(O[i],O[i+1])>=thr: cur.append(i+1)
        else: runs.append(cur); cur=[i+1]
    runs.append(cur); return runs

def repair(runs, O):
    """merge a run of len<3 with an adjacent run when the union has a size-3 core
       and the merged size is <=3 (recovers triplets whose middle link failed)."""
    out=[]; i=0
    while i<len(runs):
        r=runs[i]
        if len(r)<3 and i+1<len(runs) and len(r)+len(runs[i+1])<=3:
            cand=r+runs[i+1]
            core=set.intersection(*[O[k] for k in cand])
            if len(core)==3:
                out.append(cand); i+=2; continue
        out.append(r); i+=1
    return out

# ---------------- speed model ----------------
def fit(sub, VOCAB, alpha=0.6):
    c={s:collections.Counter() for s in (1,2,3)}
    for _,r in sub.iterrows(): c[r.s3][r.ans]+=1
    P={}
    for s in (1,2,3):
        tot=sum(c[s].values())+alpha*len(VOCAB)
        P[s]={w:(c[s][w]+alpha)/tot for w in VOCAB}
    return P

def assign_block(core, k, P):
    """core: set of 3 speed-words. k = number of videos in the block (2 or 3).
       Choose an increasing s3-subset of size k and a bijection to the k videos.
       Returns list of (word, s3) in video order + marginal posteriors."""
    core=sorted(core); cand=[]
    for s3sub in itertools.combinations((1,2,3),k):
        for perm in itertools.permutations(core,k):
            ll=sum(math.log(P[s3sub[t]][perm[t]]) for t in range(k))
            cand.append((ll,s3sub,perm))
    mx=max(c[0] for c in cand); Z=sum(math.exp(c[0]-mx) for c in cand)
    best=max(cand,key=lambda c:c[0])
    post=[]
    for t in range(k):
        p=sum(math.exp(c[0]-mx) for c in cand if c[2][t]==best[2][t])/Z
        post.append(p)
    return list(best[2]), list(best[1]), post

# ==================== TRAIN VALIDATION ====================
if __name__=='__main__':
    tr,te,hau=load()
    em=hau[hau.category=='emotion'].copy(); em['ans']=anstext(em)
    em['gid']=list(zip(em.user,em.s1,em.s2))
    em=em.sort_values('path').reset_index(drop=True)
    em['opts']=[frozenset([r.A,r.B,r.C,r.D]) for _,r in em.iterrows()]
    VOCAB=sorted(set().union(*[set(o) for o in em.opts]))

    # segment per user (users are contiguous; in test the whole block is segmented at once,
    # but a user boundary is also a group boundary so results are identical)
    O=[set(o) for o in em.opts]
    runs_all=[]; base=0
    for u,d in em.groupby('user',sort=False):
        idx=list(d.index); Ou=[O[i] for i in idx]
        rr=repair(segment_runs(Ou),Ou)
        for r in rr: runs_all.append([idx[k] for k in r])
    print('TRAIN segmentation: %d blocks  spectrum %s   (truth %s)'%(
        len(runs_all),dict(collections.Counter(len(r) for r in runs_all)),
        dict(collections.Counter(em.groupby('gid').size()))))
    truth_blocks={tuple(sorted(d.index)) for _,d in em.groupby('gid')}
    pred_blocks={tuple(sorted(r)) for r in runs_all}
    print('  exact block match: %d/%d (%.4f)'%(len(pred_blocks&truth_blocks),len(truth_blocks),
                                               len(pred_blocks&truth_blocks)/len(truth_blocks)))

    # LOO by TRUE group over the predicted blocks
    gid=list(em.gid); res=[]
    for blk in runs_all:
        gs=set(gid[i] for i in blk)
        sub=em[~em.gid.isin(gs)]
        P=fit(sub,VOCAB)
        Ob=[O[i] for i in blk]; core=set.intersection(*Ob)
        k=len(blk)
        if len(core)==3 and k in (2,3):
            words,s3s,post=assign_block(core,k,P)
            for t,i in enumerate(blk):
                res.append(dict(i=i,path=em.path[i],true_s3=em.s3[i],pred_s3=s3s[t],
                                true=em.ans[i],pred=words[t],post=post[t],mode='core%d'%k,blk=len(blk)))
        else:
            for t,i in enumerate(blk):
                r=em.loc[i]
                w=max([r.A,r.B,r.C,r.D],key=lambda x:max(P[s][x] for s in (1,2,3)))
                res.append(dict(i=i,path=r.path,true_s3=r.s3,pred_s3=np.nan,true=r.ans,pred=w,
                                post=np.nan,mode='fallback',blk=len(blk)))
    R=pd.DataFrame(res); R['ok']=R.true==R.pred
    print()
    print('=== TRAIN end-to-end (segment -> core -> assign), LOO speed model ===')
    print('  ANSWER accuracy overall: %.4f (%d/%d)'%(R.ok.mean(),R.ok.sum(),len(R)))
    print('  by mode:'); print(R.groupby('mode').ok.agg(['count','sum','mean']).to_string())
    m=R[R['mode']=='core3']
    print('  s3 recovery (core3 blocks): %.4f'%(m.pred_s3==m.true_s3).mean())
    print()
    R['bin']=pd.cut(R.post,[0,.5,.7,.85,.95,.99,1.001])
    print('  CALIBRATION (posterior vs realised accuracy):')
    print(R.dropna(subset=['post']).groupby('bin',observed=True).ok.agg(['count','mean']).round(4).to_string())
    R.to_csv('train_pipeline.csv',index=False)
