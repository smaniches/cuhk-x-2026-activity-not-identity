import itertools,math
import numpy as np
import relative_pace_today as RP
C=RP.C;E=RP.E
def infer(g,model,slots,weight,base,locked):
    n=len(g);core=set.intersection(*(set(C.option_set(r)) for r in g))
    if n not in [2,3] or len(core)!=3 or not all(RP.dur(r) for r in g):return []
    coef,sigma,_=model;scores=[]
    for words in itertools.permutations(sorted(core),n):
        letters=[next(k for k in C.live_options(r) if r[k]==w) for r,w in zip(g,words)]
        if any(r['qa_id'] in locked and a!=base[r['qa_id']] for r,a in zip(g,letters)):continue
        errors=[(math.log(RP.dur(g[i])/RP.dur(g[j]))-coef[RP.vi[words[i]]]+coef[RP.vi[words[j]]])/sigma for i,j in itertools.combinations(range(n),2)]
        score=-.5*sum(e*e for e in errors)/(n-1)
        if n==3:score+=weight*sum(E._logp(slots,w,i+1,3) for i,w in enumerate(words))
        scores.append((score,letters))
    if not scores:return []
    scores.sort(key=lambda x:(-x[0],x[1]));pp=np.exp(np.array([s for s,_ in scores])-scores[0][0]);pp/=pp.sum()
    out=[]
    for j,(r,a) in enumerate(zip(g,scores[0][1])):
        conf=float(sum(p for p,(_,aa) in zip(pp,scores) if aa[j]==a))
        out.append(dict(qa_id=r['qa_id'],current_answer=base[r['qa_id']],proposed_answer=a,
                        confidence=conf,group_size=n,subject=r['path'].split('/')[1],
                        correct=a==r.get('answer'),regression=base[r['qa_id']]==r.get('answer'),
                        family='C_relative_duration',support=len(slots.get(r[a],{})),conflict_status='none'))
    return out

def group_rows(rows):
    by=C.group_by_video(rows)
    return [[r for v in g for r in by[v] if r['category']=='emotion'] for g in C.segment_siblings(rows)]

def select(groups,threshold):
    result=[]
    for g in groups:
        if not g:continue
        # Change entire joint assignment only when each proposed new row passes.
        edits=[r for r in g if r['proposed_answer']!=r['current_answer']]
        if edits and all(r['confidence']>=threshold for r in edits):result.extend(edits)
    return result
