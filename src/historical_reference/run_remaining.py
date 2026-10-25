from pathlib import Path
import sys,csv,json,hashlib,re,collections
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'source_snapshot/cuhk_final/structural_decoder'))
import common as C
import precedence_live as D
import c2_live as F
HIST=Path('/PARTICIPANT_WORKSPACE/cuhk_large_takeover')
DATA=HERE.parent/'recovered_generators/execution'
rows=C.read_qa(DATA/'test_qa.csv');train=C.read_qa(DATA/'training_qa.csv')
def load(p):return {r['qa_id']:r['prediction'] for r in C.read_qa(p)}
def save(name,pred,lf=False):
 p=HERE/name
 with p.open('w',newline='') as f:
  w=csv.writer(f,lineterminator='\n' if lf else '\r\n');w.writerow(['qa_id','prediction']);w.writerows((r['qa_id'],pred[r['qa_id']]) for r in rows)
 return p
def check(label,p,target):
 a=load(p);b=load(target);sha=hashlib.sha256(p.read_bytes()).hexdigest();exact=sha==hashlib.sha256(target.read_bytes()).hexdigest()
 print(f'{label}_MATCH={sum(a.get(q)==v for q,v in b.items())}/682\n{label}_SHA256={sha}\n{label}_EXACT={"YES" if exact else "NO"}',flush=True)
 if not exact:
  print('FIRST_DIFFERENCE',next(((q,a.get(q),v) for q,v in b.items() if a.get(q)!=v),'serialization only'),flush=True);raise SystemExit(1)
 return sha
pred=load(HERE/'generated_candidate_C_relative_pace.csv');before=dict(pred)
edges=D.fit([r for r in train if r['category']=='sequence'],3)
for r in rows:
 if r['category']=='sequence':
  z=D.infer(r,pred[r['qa_id']],edges)
  if z and z[1]==1:pred[r['qa_id']]=z[0]
p=save('generated_candidate_D_precedence.csv',pred);check('PRECEDENCE',p,HIST/'candidate_D_precedence.csv')
# Banked-repair IDs are derived from freshly generated stage differences, not historical edit tables.
locked={e['qa_id'] for e in json.loads((HERE/'generated_boolean_edits.json').read_text())}
locked.update(e['qa_id'] for e in json.loads((HERE/'generated_joint_edits.json').read_text()))
joint=load(HERE/'generated_candidate_B_joint_siblings.csv')
locked.update(q for q in pred if pred[q]!=joint[q])
model=F.RP.fit(F.RP.er);slots=F.E.fit_slot_table(F.RP.er)
groups=[F.infer(g,model,slots,0,pred,locked) for g in F.group_rows(rows)]
edits=F.select([g for g in groups if g and g[0]['group_size']==2],.4)
for e in edits:pred[e['qa_id']]=e['proposed_answer']
p=save('generated_candidate_F_C2_relative.csv',pred);check('F_C2',p,HIST/'candidate_F_C2_relative.csv')
# Port the historical sequence decoder to generated input and the existing QA files.
t=(HIST/'attack/seqlib.py').read_text().replace('/PARTICIPANT_WORKSPACE/cuhk_work/large/',str(DATA)+'/').replace(str(HIST/'candidate_F_C2_relative.csv'),str(p))
(HERE/'seqlib_live.py').write_text(t)
import seqlib_live as S
p2,classes=S.loadP(HIST/'attack/post_win_W2.0_S1.0.npz');p3,_=S.loadP(HIST/'attack/post_win_W3.0_S1.0.npz')
answers=S.Decoder([p2,p3],classes,power=2.0,pool=True).run(S.St,1.0,test=True)
for q,(a,margin) in answers.items():
 if margin>=5:pred[q]=a
p=save('generated_candidate_L1_seq_margin5.csv',pred,True);check('L1',p,HIST/'candidate_L1_seq_margin5.csv')
def key(r):return r['source'],r['category'],tuple(sorted(r[k] for k in 'ABCD' if r.get(k)))
def subject(r):return re.search(r'user\d+',r['path']).group()
index=collections.defaultdict(list)
for r in train:
 if r['source']=='HARn':index[key(r)].append(r)
for r in rows:
 if r['source']!='HARn':continue
 group=index[key(r)];labels={x[x['answer']] for x in group};users={subject(x) for x in group}
 if len(labels)==1 and len(users)>=2:
  word=next(iter(labels));letters=[k for k in 'ABCD' if r.get(k)==word];assert len(letters)==1;pred[r['qa_id']]=letters[0]
p=save('generated_primary_A.csv',pred,True);check('PRIMARY',p,Path('/PARTICIPANT_WORKSPACE/cuhk_breakthrough_0915/CUHK_A_sha8e40fdfa92d8.csv'))
print('PRIMARY_PATH='+str(p),flush=True)
