import sys,pathlib,collections,csv,json,math
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent));from constraint_today import C,P,R

def scores(rows):
 by=C.group_by_video(rows);out={}
 for g in C.segment_siblings(rows):
  rs=[r for v in g for r in by[v]];cnt,slots,_,_=C.group_action_counts(rs);threshold=max(3,round(.09*slots))
  for r in rs:
   def score(ans):
    if r['category']=='multi':return sum((1 if k in ans else -1)*(cnt[r[k]]-threshold) for k in C.live_options(r))
    return sum(-math.log1p(math.exp(-(cnt[a]-threshold))) for a in C.option_actions(r,ans))
   out[r['qa_id']]=(r,score)
   # Bind per-group values explicitly instead of retaining closure loop variables.
   out[r['qa_id']]=(r,dict(cnt),threshold)
 return out

def choose(entry,answers):
 r,cnt,threshold=entry
 def score(ans):
  if r['category']=='multi':return sum((1 if k in ans else -1)*(cnt.get(r[k],0)-threshold) for k in C.live_options(r))
  return sum(-math.log1p(math.exp(-(cnt.get(a,0)-threshold))) for a in C.option_actions(r,ans))
 return max(answers,key=score)

def main():
 tr=C.read_qa(P/'zero_cost_pipeline/cuhkx_large_data/training_qa.csv');val=json.loads((R/'results/constraint_validation_today.json').read_text());oof={(r['protocol'],r['qa_id']):r for r in C.read_qa(R/'results/oof_rows.csv')};reports=[]
 for mode in ['full','pairs']:
  rows=[r for r in tr if mode=='full' or not r['path'].endswith('-3')];sc=scores(rows);v=[r for r in val if r['mode']==mode and r['status']=='scored'];wrong=[r for r in v if not r['gold_allowed']];changes=[]
  for r in v:
   q=r['qa_id'];b=oof.get(('full',q))
   if b and b['prediction'] not in r['possible_answers']:
    pred=choose(sc[q],r['possible_answers']);gold=sc[q][0]['answer'];changes.append(dict(qa_id=q,subject=r['subject'],correct=pred==gold,base_correct=b['prediction']==gold))
  reports.append(dict(mode=mode,rows=len(v),gold_excluded=len(wrong),changes=len(changes),correct=sum(x['correct'] for x in changes),regressions=sum(x['base_correct'] for x in changes),subjects=len({x['subject'] for x in changes}),note='Full OOF surrogate used as reference; pairs baseline not matched to this omission. Exclusion audit itself uses no baseline.'))
 te=C.read_qa(P/'zero_cost_pipeline/cuhkx_large_data/test_qa.csv');sc=scores(te);edits=json.loads((R/'results/constraint_today.json').read_text())
 for e in edits:e['proposed_answer']=choose(sc[e['qa_id']],e['possible_answers'])
 out={'validation':reports,'edits':edits};(R/'results/constraint_rank_today.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
