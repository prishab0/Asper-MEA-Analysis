import sys, json, numpy as np, multiprocessing as mp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from net import simulate
C=json.load(open('calib.json'))
ARMS={'rescue':(1.5,1.0,0.0),'supp_drive':(1.7,0.6,C['drive']),'supp_gee':(C['gee'],0.6,0.0),
      'supp_gie':(1.7,C['gie'],0.0),'disease':(1.7,0.6,0.0)}
T=30.0
def one(args):
    arm,k=args
    g_ee,g_ie,drive=ARMS[arm]
    rng=np.random.default_rng(1000*list(ARMS).index(arm)+k)
    g_ee*=float(np.exp(rng.normal(0,0.04))); g_ie*=float(np.exp(rng.normal(0,0.06))); drive+=float(rng.normal(0,0.3))
    t,i,n=simulate(g_ee,g_ie,drive,T=T,seed=int(rng.integers(1e6)))
    pr,_=np.histogram(t,bins=np.arange(0,T+0.005,0.005)); pr=gaussian_filter1d(pr/0.005/n,3)
    pk,_=find_peaks(pr,height=10,distance=60,prominence=6)
    pk=[p for p in pk if 50<=p<len(pr)-50]
    feats=[];wav=[]
    for p in pk:
        h=pr[p]; lo=p
        while lo>0 and pr[lo]>h/2: lo-=1
        hi=p
        while hi<len(pr)-1 and pr[hi]>h/2: hi+=1
        feats.append((h,(hi-lo)*5.0,(p-lo)*5.0)); wav.append(pr[p-50:p+50]/h)
    rec=dict(arm=arm,k=k,rate=len(t)/T/n,nb=len(pk))
    if len(pk)>=3:
        F=np.array(feats).mean(0); rec.update(peak=F[0],fwhm=F[1],rise=F[2],wave=np.mean(wav,0).tolist())
    return rec
if __name__=='__main__':
    N=int(sys.argv[1]) if len(sys.argv)>1 else 24
    jobs=[(a,k) for k in range(N) for a in ARMS]
    with mp.Pool(2) as pool:
        res=[]
        for r in pool.imap_unordered(one,jobs):
            res.append(r)
            if len(res)%10==0: print(len(res),'/',len(jobs)); sys.stdout.flush()
    json.dump(res,open('wells.json','w'))
    print('saved')
