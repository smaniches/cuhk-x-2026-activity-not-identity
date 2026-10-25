import collections,itertools
import common as C
def fit(rr,n):
 d=collections.defaultdict(lambda:collections.defaultdict(set))
 for r in rr:
  words=[r[k] for k in r['answer']]
  for a,b in itertools.combinations(words,2):d[tuple(sorted((a,b)))][(a,b)].add(r['path'].split('/')[1])
 return {next(iter(v)):len(next(iter(v.values()))) for v in d.values() if len(v)==1 and len(next(iter(v.values())))>=n}

def infer(r,base,edges):
 opts={r[k]:k for k in C.live_options(r)};ee=[(opts[a],opts[b],n) for (a,b),n in edges.items() if a in opts and b in opts];possible=[''.join(p) for p in itertools.permutations(C.live_options(r)) if all(p.index(a)<p.index(b) for a,b,n in ee)]
 if not possible or base in possible:return None
 def dist(p):return sum((p.index(a)<p.index(b))!=(base.index(a)<base.index(b)) for a,b in itertools.combinations(base,2))
 ds=[dist(p) for p in possible];best=[p for p,d in zip(possible,ds) if d==min(ds)]
 if len(best)!=1:return None
 return best[0],len(possible),min(n for a,b,n in ee),ee
