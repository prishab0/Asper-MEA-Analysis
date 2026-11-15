"""Robustness sweep: same rescue-vs-suppression question as run_wells.py, but with realistic well-to-well
heterogeneity. Parameter jitter is multiplied by SCALE (1 = original run)."""
import sys, json, numpy as np, multiprocessing as mp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from net import simulate
C=json.load(open('calib.json'))
ARMS={'rescue':(1.5,1.0,0.0),'supp_drive':(1.7,0.6,C['drive']),'supp_gee':(C['gee'],0.6,0.0),'supp_gie':(1.7,C['gie'],0.0)}
T=30.0
def one(args):
    arm,k,scale=args
    g_ee,g_ie,drive=ARMS[arm]
    rng=np.random.default_rng(7000+100*list(ARMS).index(arm)+k+int(scale*1000))
    g_ee*=float(np.exp(rng.normal(0,0.04*scale))); g_ie*=float(np.exp(rng.normal(0,0.06*scale))); drive+=float(rng.normal(0,0.3*scale))
    t,i,n=simulate(g_ee,g_ie,drive,T=T,seed=int(rng.integers(1e6)))
    pr,_=np.histogram(t,bins=np.arange(0,T+0.005,0.005)); pr=gaussian_filter1d(pr/0.005/n,3)
    pk,_=find_peaks(pr,height=10,distance=60,prominence=6)
    pk=[p for p in pk if 50<=p<len(pr)-50]
    rec=dict(arm=arm,k=k,scale=scale,rate=len(t)/T/n,nb=len(pk))
    if len(pk)>=3:
        W=np.mean([pr[p-50:p+50]/pr[p] for p in pk],0); peak=float(np.mean([pr[p] for p in pk]))
        base=float(np.median(W)); half=base+(1-base)/2; lo=50
        while lo>0 and W[lo]>half: lo-=1
        hi=50
        while hi<99 and W[hi]>half: hi+=1
        rec.update(peak=peak,base=base,hw=(hi-lo)*5.0,hrise=(50-lo)*5.0)
    return rec
if __name__=='__main__':
    N=int(sys.argv[1]); scales=[float(s) for s in sys.argv[2:]]
    jobs=[(a,k,s) for s in scales for k in range(N) for a in ARMS]
    out=[]
    with mp.Pool(2) as pool:
        for r in pool.imap_unordered(one,jobs):
            out.append(r)
            if len(out)%12==0: print(len(out),'/',len(jobs)); sys.stdout.flush()
    json.dump(out,open('sweep.json','w')); print('saved')
