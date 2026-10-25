import numpy as np, pandas as pd, itertools, re, collections
tr=pd.read_csv('/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution/training_qa.csv'); te=pd.read_csv('/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/recovered_generators/execution/test_qa.csv')
part=pd.read_csv('/PARTICIPANT_WORKSPACE/cuhk_large_takeover/results/test_partition_today.csv').set_index('qa_id')
champ=pd.read_csv('/PARTICIPANT_WORKSPACE/cuhkx_verification_final_20260917/reconstruction_candidate/downstream_live/generated_candidate_F_C2_relative.csv').set_index('qa_id').prediction
def loadP(f):
    d=np.load(f,allow_pickle=True); return {u:p for u,p in zip(d['units'],d['P'])}, list(d['classes'])
def user_of(p):
    m=re.search(r'user(\d+)',p); return f'user{m.group(1)}' if m else 'test'
def rows(df,is_test):
    s=df[df.category=='sequence'].copy()
    s['unit']=('Testing/'+s.path.str.split('/').str[:2].str.join('/')) if is_test else ('Training/'+s.path)
    s['user']=s.path.map(user_of)
    s['group']=s.qa_id.map(lambda q: part.loc[q,'group']) if is_test else s.path.str.rsplit('-',n=1).str[0]
    return s
S=rows(tr,False); St=rows(te,True)
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
class Decoder:
    def __init__(self,Ps,classes,power=2.0,pool=True):
        self.Ps=Ps; self.ci={c:i for i,c in enumerate(classes)}; self.power=power; self.pool=pool
    def unitP(self,u,present):
        Ps=[P[u] for P in self.Ps]; a=Ps[0]
        for b in Ps[1:]:
            idx=np.linspace(0,len(b)-1,len(a)).round().astype(int); a=a+b[idx]
        a=a/len(Ps); idx=[self.ci[x] for x in present]; Q=np.zeros_like(a); sub=a[:,idx]**self.power; Q[:,idx]=sub/(sub.sum(1,keepdims=True)+1e-9); return Q
    @staticmethod
    def pair_logit(P,i,j):
        pa=P[:,i]; pb=P[:,j]; cb=np.cumsum(pb); tot=cb[-1]
        return np.log((np.sum(pa*(tot-cb))+1e-6)/(np.sum(pa*(cb-pb))+1e-6))
    def run(self,S,ws,test=False,wp=1.0):
        preds={}
        for g,d in S.groupby('group'):
            present=sorted(set(d[['A','B','C','D']].values.ravel()))
            uP={u:self.unitP(u,present) for u in d.unit.unique()}
            cache={}
            def sf(unit,a,b):
                key=(unit if not self.pool else 'pool',a,b)
                if key not in cache:
                    if self.pool: cache[key]=sum(self.pair_logit(uP[u2],self.ci[a],self.ci[b]) for u2 in uP)/len(uP)
                    else: cache[key]=self.pair_logit(uP[unit],self.ci[a],self.ci[b])
                return cache[key]
            for _,r in d.iterrows():
                opts=[r.A,r.B,r.C,r.D]; pl=prior_logit(None if test else r.user)
                sc=[]
                for perm in PERMS:
                    o=[opts[i] for i in perm]; sc.append(sum(ws*sf(r.unit,o[i],o[j])+wp*pl(o[i],o[j]) for i in range(4) for j in range(i+1,4)))
                sc=np.array(sc); k=sc.argmax(); s=np.sort(sc)
                preds[r.qa_id]=(''.join('ABCD'[i] for i in PERMS[k]), s[-1]-s[-2])
        return preds
