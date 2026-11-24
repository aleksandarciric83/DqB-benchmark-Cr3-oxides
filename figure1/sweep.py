import numpy as np, pickle, ts_d3 as T, sys
def prep(CB):
    import itertools
    F4=CB/35; F2=1+5*F4; F={0:0.0,2:49*F2,4:441*F4}; n=T.n
    T.COUL=np.zeros((n,n,n,n))
    for p,q,r,s in itertools.product(range(n),repeat=4): T.COUL[p,q,r,s]=T.coul(p,q,r,s,F)
    Hc=T.build(0.0,F); T.COUL=np.zeros((n,n,n,n)); Hcf=T.build(1.0,F); return Hc,Hcf
SZ=np.diag(T.Sz)
I32=np.where(np.isclose(SZ,1.5))[0]; I12=np.where(np.isclose(SZ,0.5))[0]
def levels(Hc,Hcf,dq):
    H=Hc+dq*Hcf
    wq=np.linalg.eigvalsh(H[np.ix_(I32,I32)]); g0=wq[0]
    Hd=(H+100*T.S2)[np.ix_(I12,I12)]; wd=np.linalg.eigvalsh(Hd)
    wd=wd[wd<200]-75  # doublets
    def grp(w,deg_unit):
        out=[];i=0
        while i<len(w):
            j=i
            while j+1<len(w) and abs(w[j+1]-w[i])<1e-6: j+=1
            out.append((w[i]-g0,j-i+1,0)); i=j+1
        return out
    q=[(a,b*4,3.75) for a,b,c in grp(wq,1)]
    d=[(a,b*2,0.75) for a,b,c in grp(wd,1)]
    groups=q+d
    return dict(T2=q[1][0],T1F=q[2][0],T1P=q[3][0],E=[g for g in d if g[1]==4][0][0],doublets=d[:4],quart=q[:4])
if __name__=='__main__':
    CB=float(sys.argv[1]) if len(sys.argv)>1 else 4.5
    Hc,Hcf=prep(CB)
    for dq in [1.0,2.0,2.5]:
        L=levels(Hc,Hcf,dq); print(dq,[(round(a,3),b) for a,b,c in L['quart']],[(round(a,3),b) for a,b,c in L['doublets']])
        x=(L['T1F']-L['T2'])/dq; print('  eq2 Dq/B=',15*(x-8)/(x*x-10*x))
    f=lambda dq: levels(Hc,Hcf,dq)['T2']-levels(Hc,Hcf,dq)['E']
    a,b=1.5,3.0
    for _ in range(50):
        m=(a+b)/2
        if f(a)*f(m)<=0: b=m
        else: a=m
    print('C/B',CB,'crossing Dq/B=',round((a+b)/2,4))
