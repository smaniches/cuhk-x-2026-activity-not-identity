import pandas as pd, numpy as np, collections, itertools, math, os
from base import load, anstext
from pipeline import segment_runs, repair, fit, jac, THR
D='/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution'
tr,te,hau=load()
em=hau[hau.category=='emotion'].copy(); em['ans']=anstext(em)
em['gid']=list(zip(em.user,em.s1,em.s2)); em=em.sort_values('path').reset_index(drop=True)
em['opts']=[set([r.A,r.B,r.C,r.D]) for _,r in em.iterrows()]
tem=te[te.category=='emotion'].copy().sort_values('pid').reset_index(drop=True)
tem['opts']=[set([r.A,r.B,r.C,r.D]) for _,r in tem.iterrows()]
VOCAB=sorted(set().union(*list(em.opts))|set().union(*list(tem.opts)))

def triplet_assign(optlists, P, core=None):
    """choose distinct (w1,w2,w3), wt in optlists[t], s3=t+1, maximising sum log P."""
    if core is not None and len(core)==3:
        pools=[sorted(core)]*3
    else:
        pools=[sorted(o) for o in optlists]
    cands=[]
    for combo in itertools.product(*pools):
        if len(set(combo))!=3: continue
        cands.append((sum(math.log(P[t+1][combo[t]]) for t in range(3)),combo))
    mx=max(c[0] for c in cands); Z=sum(math.exp(c[0]-mx) for c in cands)
    best=max(cands,key=lambda c:c[0])[1]
    post=[sum(math.exp(c[0]-mx) for c in cands if c[1][t]==best[t])/Z for t in range(3)]
    return list(best),post

# ---------- TRAIN LOO validation of the exact rule used for the test ----------
res=[]
for g,d in em.groupby('gid'):
    if len(d)!=3: continue
    sub=em[em.gid!=g]; P=fit(sub,VOCAB)
    d=d.sort_values('s3'); ol=list(d.opts); core=set.intersection(*ol)
    w,post=triplet_assign(ol,P,core if len(core)==3 else None)
    for t,(_,r) in enumerate(d.iterrows()):
        res.append(dict(gid=g,ok=w[t]==r.ans,post=post[t],mode='core' if len(core)==3 else 'broken'))
R=pd.DataFrame(res)
print('=== TRAIN LOO: rule actually applied to the test (triplets, TRUE s3 known) ===')
print(R.groupby('mode').ok.agg(['count','sum','mean']).to_string())
print('  OVERALL: %.4f (%d/%d)'%(R.ok.mean(),R.ok.sum(),len(R)))
CAL=R.ok.mean()/R.post.mean()
print('  mean posterior %.4f  realised %.4f  calibration ratio %.4f'%(R.post.mean(),R.ok.mean(),CAL))
R['bin']=pd.cut(R.post,[0,.5,.7,.85,.95,.99,1.001])
print(R.groupby('bin',observed=True).ok.agg(['count','mean']).round(4).to_string())

# ---------- BUILD TEST PREDICTIONS ----------
P=fit(em,VOCAB)
champ=pd.read_csv(os.path.join(D,'cand_emotion_s3_v1.csv')); cp=dict(zip(champ.qa_id,champ.prediction))
pid=list(tem.pid)
rows=[]
# clean region: pids 65..163 = 33 triplets, phase = ((pid-65)%3)+1
for start in range(65,164,3):
    d=tem[tem.pid.isin([start,start+1,start+2])].sort_values('pid')
    ol=list(d.opts); core=set.intersection(*ol)
    w,post=triplet_assign(ol,P,core if len(core)==3 else None)
    for t,(_,r) in enumerate(d.iterrows()):
        opts={c:r[c] for c in 'ABCD'}
        cw=opts.get(cp[r.qa_id])
        # posterior that the CHAMPION's word is right, from the same joint model
        pools=[sorted(core)]*3 if len(core)==3 else [sorted(o) for o in ol]
        cands=[(sum(math.log(P[u+1][cb[u]]) for u in range(3)),cb)
               for cb in itertools.product(*pools) if len(set(cb))==3]
        mx=max(c[0] for c in cands); Z=sum(math.exp(c[0]-mx) for c in cands)
        pc=sum(math.exp(c[0]-mx) for c in cands if c[1][t]==cw)/Z
        rows.append(dict(qa_id=r.qa_id,pid=r.pid,region='clean',s3=t+1,
            block=start,core='|'.join(sorted(core)) if len(core)==3 else '(broken)',
            champ_letter=cp[r.qa_id],champ_word=cw,P_champ=pc,
            new_word=w[t],new_letter=[c for c in 'ABCD' if r[c]==w[t]][0],P_new=post[t],
            delta_raw=post[t]-pc))
# messy region 164..208: leave as champion
for _,r in tem[tem.pid>=164].iterrows():
    opts={c:r[c] for c in 'ABCD'}
    rows.append(dict(qa_id=r.qa_id,pid=r.pid,region='pairs',s3=np.nan,block=np.nan,core='',
        champ_letter=cp[r.qa_id],champ_word=opts.get(cp[r.qa_id]),P_champ=np.nan,
        new_word=opts.get(cp[r.qa_id]),new_letter=cp[r.qa_id],P_new=np.nan,delta_raw=0.0))
C=pd.DataFrame(rows)
C['changed']=C.champ_letter!=C.new_letter
C['delta_cal']=C.delta_raw*CAL
C.to_csv('ranked_candidates.csv',index=False)
print()
print('=== TEST clean region (pids 65-163, 33 triplets, 99 rows) ===')
cl=C[C.region=='clean']
print('  disagreements with champion: %d / 99'%cl.changed.sum())
print('  sum P(champion correct) = %.2f   sum P(new correct) = %.2f'%(cl.P_champ.sum(),cl.P_new.sum()))
print('  RAW expected delta      = %+.2f items'%cl.delta_raw.sum())
print('  calibration-adjusted    = %+.2f items  (x%.3f train ratio)'%(cl.delta_raw.sum()*CAL,CAL))
print()
print('  delta by posterior tier of the NEW pick:')
cl2=cl[cl.changed].copy(); cl2['tier']=pd.cut(cl2.P_new,[0,.5,.7,.85,.95,.99,1.001])
print(cl2.groupby('tier',observed=True).agg(n=('qa_id','size'),sumPnew=('P_new','sum'),
      sumPchamp=('P_champ','sum'),delta=('delta_raw','sum')).round(2).to_string())
