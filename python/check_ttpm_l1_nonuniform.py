"""Independent stress checks for the L1-kernel TTPM comparison.

Run with a Python interpreter providing NumPy. The fixed seed exercises
nonuniform time grids, analytic kernel-cell integrals, and Lipschitz
impact laws. The odd-sine case additionally checks the estimate beyond
the theorem's sign-preserving impact class. Floating-point sanity checks
supplement the analytic proof in no_manipulation_v2.tex.
"""

import numpy as np

rng=np.random.default_rng(442891)

def cap(x,widths,target):
    if target <= 0:
        return np.zeros_like(x)
    if target >= widths@x:
        return x.copy()
    lo,hi=0.,float(x.max())
    for _ in range(62):
        mid=(lo+hi)/2
        if widths@np.minimum(x,mid) < target: lo=mid
        else: hi=mid
    return np.minimum(x,(lo+hi)/2)

def kernel_weights(edges,kind):
    n=len(edges)-1
    if kind=='delayed':
        l=np.array([rng.uniform(.03,.85)])
        r=np.minimum(1.,l+rng.uniform(.001,.2))
        z=np.array([rng.lognormal(0,2)])
    elif kind=='multipeak':
        l=rng.uniform(0,.98,8)
        r=np.minimum(1.,l+rng.uniform(.001,.18,8))
        z=rng.lognormal(0,2,8)
    elif kind=='near_diagonal':
        l=np.array([0.])
        r=np.array([rng.uniform(.002,.15)])
        z=np.array([1./r[0]])
    elif kind=='oscillating':
        l=np.arange(0,1,.05)
        r=l+.025
        z=rng.lognormal(0,1,len(l))
    elif kind=='permanent':
        l=np.array([0.]);r=np.array([1.]);z=np.array([1.])
    else: raise ValueError(kind)
    def F(x):
        return np.sum(z*(np.maximum(np.asarray(x)[...,None]-l,0)-np.maximum(np.asarray(x)[...,None]-r,0)),axis=-1)
    def Psi(x):
        return .5*np.sum(z*(np.maximum(np.asarray(x)[...,None]-l,0)**2-np.maximum(np.asarray(x)[...,None]-r,0)**2),axis=-1)
    ai=edges[:-1,None];bi=edges[1:,None]
    aj=edges[None,:-1];bj=edges[None,1:]
    W=np.tril(Psi(bi-aj)-Psi(ai-aj)-Psi(bi-bj)+Psi(ai-bj),-1)
    np.fill_diagonal(W,Psi(np.diff(edges)))
    # Cancellation can leave tiny negative roundoff in exact zero weights.
    assert W.min()>-2e-12*(1+z.sum())
    W=np.maximum(W,0)
    pts=np.unique(np.r_[0,1,l,r,1-l,1-r])
    R=float(np.max(F(pts)+F(1-pts)))
    return W,R

laws=[
 ('sine_nonmonotone',lambda x:x+.55*np.sin(5*x),3.75),
 ('rational_decay',lambda x:x/(1+x*x),1.),
 ('tent_plus_linear',lambda x:.01*x+np.sign(x)*np.maximum(0,1-np.abs(np.abs(x)-1)),1.01),
 ('odd_sine',np.sin,1.),
 ('linear',lambda x:x,1.),
 ('tanh',np.tanh,1.),
]
kinds=['delayed','multipeak','near_diagonal','oscillating','permanent']
maxratio=-1.;worst=None;maxvol=0.;maxpathvol=0.;count=0;pathchecks=0
bylaw={name:0. for name,_,_ in laws};bykernel={k:0. for k in kinds}
for case in range(12000):
    n=int(rng.choice([2,3,8,16,32,64]))
    widths=rng.dirichlet(np.full(n,rng.choice([.3,1.,5.])))
    # Exclude vanishing widths where cancellation swamps the exact-cell integral.
    widths=np.maximum(widths,1e-7);widths/=widths.sum()
    edges=np.r_[0,np.cumsum(widths)];edges[-1]=1
    mode=case%5
    if mode==0: v=rng.normal(size=n)
    elif mode==1:v=rng.choice([-1.,1.],n)*rng.lognormal(0,2,n)
    elif mode==2:v=np.where(np.arange(n)%2,1.,-1.)*rng.lognormal(0,.5,n)
    elif mode==3:
        v=rng.uniform(.1,1,n);v[rng.integers(n)]=-10.**rng.uniform(-10,1)
    else:
        v=-rng.uniform(.001,1,n);v[rng.integers(n)]=10.**rng.uniform(0,5)
    v*=10.**rng.uniform(-3,3)
    if widths@v <0:v=-v
    Q=float(widths@v)
    if Q<=1e-15 or not np.any(v<0):continue
    p=np.maximum(v,0);nminus=np.maximum(-v,0)
    P=float(widths@p);B=float(widths@nminus)
    u=cap(p,widths,Q)
    D=float(widths@(v*v-u*u));H=float(widths@np.abs(v-u))
    assert D>0 and np.min(u)>=0
    volerr=abs(float(widths@u)-Q)/(1+P+B)
    maxvol=max(maxvol,volerr)
    assert volerr <2e-13
    assert abs(H-2*B)<5e-13*(1+P+B)
    kind=kinds[case%len(kinds)];W,R=kernel_weights(edges,kind)
    for name,f,L in laws:
        Jv=float(v@W@f(v));Ju=float(u@W@f(u))
        rhs=L*R*D
        ratio=abs(Jv-Ju)/rhs
        count+=1
        bylaw[name]=max(bylaw[name],ratio);bykernel[kind]=max(bykernel[kind],ratio)
        if ratio>maxratio:maxratio=ratio;worst=(case,name,kind,len(v),Q,D)
        # Tiny cancellations when the removed volume is tiny are judged absolutely.
        assert abs(Jv-Ju) <= rhs+2e-11*(1+abs(Jv)+abs(Ju)),(ratio,worst)
        # At the critical eta and arbitrary nonnegative spread, cost cannot increase.
        sigma=10.**rng.uniform(-4,3)
        delta=rhs+sigma*H+Jv-Ju
        assert delta>=-2e-11*(1+abs(Jv)+abs(Ju)),(delta,worst)
    if case%7==0:
        for theta in [.001,.13,.6,.99,1.]:
            s=theta*B
            w=cap(p,widths,P-s)-cap(nminus,widths,B-s)
            err=abs(float(widths@w)-Q)/(1+P+B)
            maxpathvol=max(maxpathvol,err)
            assert err<2e-13
            assert float(widths@(w*w))<=float(widths@(v*v))*(1+2e-13)
            pathchecks+=1
print({'inequality_checks':count,'intermediate_path_checks':pathchecks,'max_ratio':maxratio,'worst':worst,'max_final_volume_relative_error':maxvol,'max_intermediate_volume_relative_error':maxpathvol,'law_maxima':bylaw,'kernel_maxima':bykernel})
