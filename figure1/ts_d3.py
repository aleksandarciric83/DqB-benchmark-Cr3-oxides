import numpy as np, itertools
from sympy.physics.wigner import wigner_3j
def ck(k,m1,m2,l=2):
    return float((-1)**m1*(2*l+1)*wigner_3j(l,k,l,0,0,0)*wigner_3j(l,k,l,-m1,m1-m2,m2))
ms=[-2,-1,0,1,2]
orbs=[(m,s) for m in ms for s in (0,1)]  # 10 spin orbitals
n=len(orbs)
CK={(k,a,b):ck(k,a,b) for k in (0,2,4) for a in ms for b in ms}
def coul(p,q,r,s,F):  # <pq|rs> with p,r electron1 ; q,s electron2
    (mp,sp),(mq,sq),(mr,sr),(ms_,ss)=orbs[p],orbs[q],orbs[r],orbs[s]
    if sp!=sr or sq!=ss or mp+mq!=mr+ms_: return 0.0
    return sum(CK[(k,mp,mr)]*CK[(k,ms_,mq)]*F[k] for k in (0,2,4))
def cf(Dq):
    V=np.zeros((5,5)); idx={m:i for i,m in enumerate(ms)}
    for m,v in {2:1,-2:1,1:-4,-1:-4,0:6}.items(): V[idx[m],idx[m]]=v*Dq
    V[idx[2],idx[-2]]=V[idx[-2],idx[2]]=5*Dq
    h=np.zeros((n,n))
    for i,(mi,si) in enumerate(orbs):
        for j,(mj,sj) in enumerate(orbs):
            if si==sj: h[i,j]=V[idx[mi],idx[mj]]
    return h
dets=[c for c in itertools.combinations(range(n),3)]
dix={d:i for i,d in enumerate(dets)}
def apply(op_list,det):
    # op_list: list of (type,orb) applied right-to-left; det sorted tuple; return (sign,newdet) or None
    occ=list(det); sign=1
    for typ,o in reversed(op_list):
        if typ=='a':
            if o not in occ: return None
            pos=occ.index(o); sign*=(-1)**pos; occ.pop(pos)
        else:
            if o in occ: return None
            pos=sum(1 for x in occ if x<o); sign*=(-1)**pos; occ.insert(pos,o)
    return sign,tuple(occ)
def build(Dq,F):
    N=len(dets); H=np.zeros((N,N)); h=cf(Dq)
    for J,d in enumerate(dets):
        for p in range(n):
            for q in range(n):
                if h[p,q]==0: continue
                r=apply([('c',p),('a',q)],d)
                if r: H[dix[r[1]],J]+=r[0]*h[p,q]
        for p,q,r_,s in itertools.product(range(n),repeat=4):
            if p==q or r_==s: continue
            v=COUL[p,q,r_,s]
            if v==0: continue
            rr=apply([('c',p),('c',q),('a',s),('a',r_)],d)
            if rr: H[dix[rr[1]],J]+=0.5*rr[0]*v
    return H
def S2(det_vecs):
    pass
B=1.0
def levels(DqB,CB):
    global COUL
    F4=CB/35*B; F2=B+5*F4
    F={0:0.0,2:49*F2,4:441*F4}
    COUL=np.zeros((n,n,n,n))
    for p,q,r,s in itertools.product(range(n),repeat=4): COUL[p,q,r,s]=coul(p,q,r,s,F)
    H=build(DqB*B,F); w,v=np.linalg.eigh(H); return w-w[0],v
# spin: Sz of each det; total S via S^2 operator
def s2mat():
    N=len(dets); M=np.zeros((N,N))
    for J,d in enumerate(dets):
        for p in range(n):
            for q in range(n):
                mp,sp=orbs[p]; mq,sq=orbs[q]
                # S+ S- etc: S^2 = S-S+ + Sz^2 + Sz
                pass
    return M
def opmat(terms):
    N=len(dets); M=np.zeros((N,N))
    for J,d in enumerate(dets):
        for coef,ops in terms:
            r=apply(ops,d)
            if r: M[dix[r[1]],J]+=coef*r[0]
    return M
up=lambda m: orbs.index((m,0)); dn=lambda m: orbs.index((m,1))
Sp=opmat([(1,[('c',up(m)),('a',dn(m))]) for m in ms])
Sm=Sp.T
Sz=np.diag([sum(0.5 if orbs[o][1]==0 else -0.5 for o in d) for d in dets])
S2=Sm@Sp+Sz@Sz+Sz
def prepare(CB):
    global COUL
    F4=CB/35; F2=1+5*F4
    F={0:0.0,2:49*F2,4:441*F4}
    COUL=np.zeros((n,n,n,n))
    for p,q,r,s in itertools.product(range(n),repeat=4): COUL[p,q,r,s]=coul(p,q,r,s,F)
    Hc=build(0.0,F)
    Hcf=build(1.0,{0:0,2:0,4:0})  # careful: build uses COUL global
    return Hc,Hcf
