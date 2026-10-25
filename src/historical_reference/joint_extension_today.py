"""Minimum-answer-change joint sibling consistency, with independent frequency tie-break."""
import pathlib,sys,json,csv,collections,math
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from constraint_today import C,P,R

def solve(rs,pred):
 acts=sorted({a for r in rs if r['category'] not in ['emotion','object_interaction'] for k in C.live_options(r) for a in C.option_actions(r,k)});ix={a:i for i,a in enumerate(acts)};n=len(acts);cons=[];comb={};obj=[0.]*n
 def var(cost=0.):
  nonlocal n
  z=n;n+=1;obj.append(cost);return z
 def add(d,lo,hi):cons.append((d,lo,hi))
 def conjunction(bits,cost=0.):
  z=var(cost)
  for a,b in bits:add({z:1,a:-1 if b else 1},-np.inf,0 if b else 1)
  dd={z:1}
  for a,b in bits:dd[a]=dd.get(a,0)+(-1 if b else 1)
  add(dd,1-sum(b for a,b in bits),np.inf)
  return z
 for r in rs:
  cat=r['category'];ks=C.live_options(r);q=r['qa_id']
  if cat=='single':add({ix[r[k]]:1 for k in ks},1,1)
  elif cat=='multi':add({ix[r[k]]:1 for k in ks},1,len(ks))
  elif cat=='sequence':
   for k in ks:add({ix[r[k]]:1},1,1)
  elif cat=='combination':
   zz=[]
   for k in ks:
    z=conjunction([(ix[a],1) for a in C.option_actions(r,k)]);comb[q,k]=z;zz.append(z)
   add({z:1 for z in zz},1,1)
  if q in pred:
   if cat in ['single','multi']:conjunction([(ix[r[k]],int(k in pred[q])) for k in ks],-1)
   elif cat=='combination':obj[comb[q,pred[q]]]=-1
 cnt,slots,_,_=C.group_action_counts(rs);thr=max(3,round(.09*slots));den=sum(abs(cnt[a]-thr) for a in acts)+1
 for a,i in ix.items():obj[i]=-.01*(cnt[a]-thr)/den
 mat=np.zeros((len(cons),n));lo=[];hi=[]
 for i,(d,l,h) in enumerate(cons):
  for j,v in d.items():mat[i,j]=v
  lo.append(l);hi.append(h)
 res=milp(obj,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(mat,lo,hi),options={'time_limit':3})
 if res.status==2:return None
 if res.status!=0:raise RuntimeError(res.message)
 out={}
 for r in rs:
  q=r['qa_id'];cat=r['category'];ks=C.live_options(r)
  if cat in ['single','multi']:out[q]=''.join(k for k in ks if res.x[ix[r[k]]]>0.5)
  elif cat=='combination':out[q]=''.join(k for k in ks if res.x[comb[q,k]]>.5)
 return out

def run(rows,pred):
 by=C.group_by_video(rows);out={};groups={}
 for i,g in enumerate(C.segment_siblings(rows)):
  rs=[r for v in g for r in by[v]];z=solve(rs,pred)
  if z is None:continue
  out.update(z);groups.update({q:(i,len(g)) for q in z})
 return out,groups

def main():
 tr=C.read_qa(P/'zero_cost_pipeline/cuhkx_large_data/training_qa.csv');oof=C.read_qa(R/'results/oof_rows.csv');report=[]
 for mode in ['full','pairs']:
  oo=[r for r in oof if r['protocol']==mode];ids={r['qa_id'] for r in oo};rr=[r for r in tr if r['qa_id'] in ids];base={r['qa_id']:r['prediction'] for r in oo};gold={r['qa_id']:r['answer'] for r in rr}
  # Candidate A stage applied first, using the already evaluated feasibility decoder.
  sys.path.insert(0,str(R/'code'));from constraint_today import possibilities
  from rank_constraint_today import scores,choose
  sc=scores(rr);by=C.group_by_video(rr)
  for g in C.segment_siblings(rr):
   rs=[r for v in g for r in by[v]];ps=possibilities(rs,rs)
   if ps:
    for q,ans in ps.items():
     if base[q] not in ans:base[q]=choose(sc[q],ans)
  new,groups=run(rr,base);changes=[dict(qa_id=q,subject=next(r['path'].split('/')[1] for r in rr if r['qa_id']==q),base=base[q],prediction=a,gold=gold[q],correct=int(a==gold[q]),regression=int(base[q]==gold[q]),group_size=groups[q][1]) for q,a in new.items() if a!=base[q]]
  report.append(dict(mode=mode,changes=len(changes),correct=sum(r['correct'] for r in changes),wrong=sum(not r['correct'] for r in changes),regressions=sum(r['regression'] for r in changes),rows=changes));print(mode, {k:v for k,v in report[-1].items() if k!='rows'},flush=True)
 te=C.read_qa(P/'zero_cost_pipeline/cuhkx_large_data/test_qa.csv');base={r['qa_id']:r['prediction'] for r in C.read_qa(R/'frozen/champion_56148432.csv')};new,groups=run(te,base);edits=[dict(qa_id=q,champion_answer=base[q],candidate_answer=a,group=groups[q]) for q,a in new.items() if a!=base[q]]
 (R/'results/joint_extension_today.json').write_text(json.dumps({'validation':report,'edits':edits},indent=2));print('TEST',edits,flush=True)
if __name__=='__main__':main()
