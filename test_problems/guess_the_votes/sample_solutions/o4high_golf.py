from itertools import product as P
def guess_the_votes(s,v):o=list(s);a=[t for t in P(v,repeat=len(o))if all(sum(s[o[i]]for i,x in enumerate(t)if x==j)==v[j]for j in v)];r={j:set()for j in v};[r[a[0][i]].add(O)for i,O in enumerate(o)if len({t[i]for t in a})==1];return r
