"""Subject-held-out pair log-duration ratios remove shared action workload."""
import pathlib,sys,csv,json,collections,itertools,math
import numpy as np
from sklearn.linear_model import Ridge
P=pathlib.Path('/PARTICIPANT_WORKSPACE');R=P/'cuhk_large_takeover';sys.path.insert(0,'/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/source_snapshot/cuhk_final/structural_decoder');import common as C,emotion_triplet as E
tr=C.read_qa(P/'zero_cost_pipeline/cuhkx_large_data/training_qa.csv');te=C.read_qa(P/'zero_cost_pipeline/cuhkx_large_data/test_qa.csv');nf=E.build_duration_index(json.loads((pathlib.Path('/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/source_snapshot/cuhk_final/multimodal/nframes.json')).read_text()));er=[r for r in tr if r['category']=='emotion'];vocab=sorted({r[k] for r in er for k in 'ABCD'});vi={w:i for i,w in enumerate(vocab)}
def dur(r):return E.lookup_duration(nf,C.video_key(r['path']))
def fit(rows):
 sessions=collections.defaultdict(list)
 for r in rows:sessions[r['path'].rsplit('-',1)[0]].append(r)
 X=[];y=[];absd=collections.defaultdict(list)
 for g in sessions.values():
  for r in g:
   if dur(r):absd[r[r['answer']]].append(math.log(dur(r)))
  for a,b in itertools.combinations(g,2):
   if not dur(a) or not dur(b):continue
   x=np.zeros(len(vocab));x[vi[a[a['answer']]]]+=1;x[vi[b[b['answer']]]]-=1;X.append(x);y.append(math.log(dur(a)/dur(b)))
 m=Ridge(alpha=2,fit_intercept=False).fit(X,y);sigma=max(float(np.std(np.array(y)-m.predict(X))),.1)
 stats={w:(np.mean(v),max(np.std(v),.3)) for w,v in absd.items()}
 return m.coef_,sigma,stats

def infer(g,model,absolute):
 core=set.intersection(*(set(C.option_set(r)) for r in g))
 if len(g)!=2 or len(core)!=3 or not all(dur(r) for r in g):return None
 coef,sigma,stats=model;delta=math.log(dur(g[0])/dur(g[1]));ss=[]
 for words in itertools.permutations(sorted(core),2):
  s=-.5*((delta-coef[vi[words[0]]]+coef[vi[words[1]]])/sigma)**2
  if absolute:
   for r,w in zip(g,words):
    mu,sd=stats.get(w,(5.3,1));s+=absolute*(-.5*((math.log(dur(r))-mu)/sd)**2-math.log(sd))
  ss.append((s,words))
 ss.sort(reverse=True);pr=np.exp(np.array([s for s,w in ss])-ss[0][0]);pr/=pr.sum()
 out=[]
 for j,r in enumerate(g):
  probs=collections.Counter()
  for pp,(_,words) in zip(pr,ss):probs[words[j]]+=pp
  word=max(probs,key=probs.get);out.append((next(k for k in 'ABCD' if r[k]==word),float(probs[word])))
 return out

def main():
 oof={(r['protocol'],r['qa_id']):r for r in C.read_qa(R/'results/oof_rows.csv')};users=sorted({r['path'].split('/')[1] for r in er});records=[]
 for u in users:
  model=fit([r for r in er if r['path'].split('/')[1]!=u]);sessions=collections.defaultdict(list)
  # Use exactly the existing OOF pair omission, allowing a matched comparison.
  for r in er:
   if r['path'].split('/')[1]==u and ('pairs',r['qa_id']) in oof:sessions[r['path'].rsplit('-',1)[0]].append(r)
  for g in sessions.values():
   for absolute in [0,1]:
    z=infer(g,model,absolute)
    if not z:continue
    for r,(pred,conf) in zip(g,z):
     b=oof['pairs',r['qa_id']]
     if b['mechanism']=='emotion_duration':continue
     records.append(dict(qa_id=r['qa_id'],subject=u,absolute=absolute,prediction=pred,base=b['prediction'],gold=r['answer'],confidence=conf,correct=pred==r['answer'],base_correct=b['prediction']==r['answer']))
 model=fit(er);by=C.group_by_video(te);champ={r['qa_id']:r['prediction'] for r in C.read_qa(R/'frozen/champion_56148432.csv')};test=[]
 for g in C.segment_siblings(te):
  if len(g)!=2:continue
  rr=[next(r for r in by[v] if r['category']=='emotion') for v in g]
  for absolute in [0,1]:
   z=infer(rr,model,absolute)
   if z:
    for r,(pred,conf) in zip(rr,z):
     if pred!=champ[r['qa_id']]:test.append(dict(qa_id=r['qa_id'],champion=champ[r['qa_id']],prediction=pred,confidence=conf,absolute=absolute))
 reports=[]
 for a in [0,1]:
  for t in [0,.6,.7,.8,.9]:
   rr=[r for r in records if r['absolute']==a and r['confidence']>=t and r['prediction']!=r['base']];ee=[r for r in test if r['absolute']==a and r['confidence']>=t]
   reports.append(dict(absolute=a,threshold=t,changes=len(rr),correct=sum(r['correct'] for r in rr),wrong=sum(not r['correct'] for r in rr),regressions=sum(r['base_correct'] for r in rr),subjects=len({r['subject'] for r in rr}),test_changes=len(ee),edits=ee))
 (R/'results/relative_pace_today.json').write_text(json.dumps({'reports':reports,'oof':records},indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
