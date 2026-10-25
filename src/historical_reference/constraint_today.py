"""Label-free Boolean action-set feasibility; no fitted thresholds."""
import pathlib,sys,json,collections,itertools
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
P=pathlib.Path('/PARTICIPANT_WORKSPACE');R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,'/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/source_snapshot/cuhk_final/structural_decoder');import common as C

def possibilities(rs,targets):
 acts=sorted({a for r in rs if r['category'] not in ['emotion','object_interaction'] for k in C.live_options(r) for a in C.option_actions(r,k)});ix={a:i for i,a in enumerate(acts)};n=len(acts);constraints=[];comb={}
 def add(d,lo,hi):constraints.append((d,lo,hi))
 for r in rs:
  cat=r['category'];ks=C.live_options(r)
  if cat=='single':add({ix[r[k]]:1 for k in ks},1,1)
  elif cat=='multi':add({ix[r[k]]:1 for k in ks},1,len(ks))
  elif cat=='sequence':
   for k in ks:add({ix[r[k]]:1},1,1)
  elif cat=='combination':
   zz=[]
   for k in ks:
    z=n;n+=1;zz.append(z);aa=[ix[a] for a in C.option_actions(r,k)];comb[r['qa_id'],k]=z
    for a in aa:add({z:1,a:-1},-np.inf,0)
    add({z:1,**{a:-1 for a in aa}},1-len(aa),np.inf)
   add({z:1 for z in zz},1,1)
 mat=np.zeros((len(constraints),n));lo=[];hi=[]
 for i,(d,l,h) in enumerate(constraints):
  for j,v in d.items():mat[i,j]=v
  lo.append(l);hi.append(h)
 lc=LinearConstraint(mat,lo,hi)
 def feasible(fixed):
  low=np.zeros(n);high=np.ones(n)
  for i,v in fixed.items():low[i]=high[i]=v
  res=milp(np.zeros(n),integrality=np.ones(n),bounds=Bounds(low,high),constraints=lc,options={'time_limit':2})
  if res.status not in [0,2]:raise RuntimeError(str(res.message))
  return res.status==0
 if not feasible({}):return None
 out={}
 for r in targets:
  cat=r['category'];ks=C.live_options(r);ans=[]
  if cat=='multi':
   for bits in itertools.product([0,1],repeat=len(ks)):
    if not any(bits):continue
    if feasible({ix[r[k]]:b for k,b in zip(ks,bits)}):ans.append(''.join(k for k,b in zip(ks,bits) if b))
  elif cat in ['single','combination']:
   for k in ks:
    if feasible({comb[r['qa_id'],k] if cat=='combination' else ix[r[k]]:1}):ans.append(k)
  else:continue
  out[r['qa_id']]=ans
 return out

def main():
 te=C.read_qa(P/'zero_cost_pipeline/cuhkx_large_data/test_qa.csv');champ={r['qa_id']:r['prediction'] for r in C.read_qa(R/'frozen/champion_56111316.csv')};by=C.group_by_video(te);records=[]
 for i,g in enumerate(C.segment_siblings(te)):
  rs=[r for v in g for r in by[v]];poss=possibilities(rs,rs)
  if poss is None:records.append(dict(group=i,status='infeasible'));continue
  for q,answers in poss.items():
   if champ[q] not in answers:records.append(dict(group=i,qa_id=q,champion=champ[q],possible_answers=answers,status='violation',forced=len(answers)==1))
 (R/'results/constraint_today.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
if __name__=='__main__':main()
