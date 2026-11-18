"""Simulate plates of wells for four arms with the slow-dynamics network (net3).
disease = reduced inhibition (g_ie 0.5), rescue = inhibition restored, supp_gee / supp_drive = disease state with a
second mechanism lowering firing to the control rate. Plate and well jitter on parameters. SIMULATED data."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from net3 import simulate
BASE=dict(base=5,U=0.02,k=0.8,tau_d=6000,tau_s=150,T=60)
ARMS={'disease':dict(g_ie=.5),'rescue':dict(g_ie=1.0),'supp_gee':dict(g_ie=.5,g_ee=.74),'supp_drive':dict(g_ie=.5,drive=-.75)}
def feats(t,n,T=60):
    pr,_=np.histogram(t,bins=np.arange(0,T+0.01,0.01)); pr=gaussian_filter1d(pr/0.01/n,5)
    base=np.median(pr); pk,_=find_peaks(pr,prominence=max(3*base,4),distance=100)
    out=dict(rate=len(t)/T/n,bursts=len(pk))
    if len(pk)<3: return out
    wd=[];rise=[];dec=[];pkh=[]
    for p in pk:
        h=pr[p]; half=base+(h-base)/2; lo=p
        while lo>0 and pr[lo]>half: lo-=1
        hi=p
        while hi<len(pr)-1 and pr[hi]>half: hi+=1
        wd.append((hi-lo)*.01); rise.append((p-lo)*.01); dec.append((hi-p)*.01); pkh.append(h)
    out.update(width=np.mean(wd),interval=np.mean(np.diff(pk))*.01,frac=np.sum(wd)/T,
               peak=np.mean(pkh),rise=np.mean(rise),asym=np.mean(dec)/max(np.mean(rise),1e-3),base=base)
    return out
def job(a):
    plate,arm,w,seed,pd_,pe,pi=a; rng=np.random.default_rng(seed)
    kw=dict(ARMS[arm]); kw['drive']=kw.get('drive',0)+pd_+rng.normal(0,.15)
    kw['g_ee']=kw.get('g_ee',1.0)*pe*np.exp(rng.normal(0,.04)); kw['g_ie']=kw.get('g_ie',1.0)*pi*np.exp(rng.normal(0,.06))
    t,i,n=simulate(seed=seed,**{**BASE,**kw})
    return dict(plate=plate,arm=arm,well=w,**feats(t,n))
if __name__=='__main__':
    rng=np.random.default_rng(1); jobs=[];s=1000
    for plate in range(4):
        pd_,pe,pi=rng.normal(0,.15),np.exp(rng.normal(0,.03)),np.exp(rng.normal(0,.04))
        arms=list(rng.permutation(np.repeat(list(ARMS),6)))
        for w,arm in enumerate(arms): s+=1; jobs.append((plate,arm,w,s,pd_,pe,pi))
    with Pool(2) as p: rows=p.map(job,jobs)
    df=pd.DataFrame(rows); df.to_csv('arms_wells.csv',index=False)
    print(len(df),'wells;',df.bursts.ge(3).sum(),'with >=3 bursts'); print(df.groupby('arm')[['rate','bursts','width','interval','peak','rise','asym']].mean().round(2))
    c=df.groupby(['plate','arm']).rate.mean().unstack().round(1); print('\nmean rate per plate and arm\n',c)
    print('\nwell-to-well rate CV within arm:',(df.groupby('arm').rate.std()/df.groupby('arm').rate.mean()).round(2).to_dict())
