def guess_the_votes(s,v):
 from itertools import product
 r={k:set()for k in v}
 a=list(s)
 o=list(v)
 A=[x for x in product(range(len(o)),repeat=len(a))if all(sum(s[a[i]]for i in range(len(a))if x[i]==j)==v[o[j]]for j in range(len(o)))]
 for i,n in enumerate(a):
  p={o[x[i]]for x in A}
  if len(p)==1:r[p.pop()].add(n)
 return r
