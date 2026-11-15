import sys, numpy as np
from tune3 import stats
from net3 import simulate
P=dict(g_ee=1,base=5,U=0.02,k=0.8,tau_d=6000,tau_s=150,T=90)
print('arm -> rate, bursts/90s, width s, interval s, frac (mean of 4 seeds, sd of bursts)')
for name,kw in [('control',{}),('GABA block 50%',dict(g_ie=0.5)),('GABA block 90%',dict(g_ie=0.1)),('drive -1 mV',dict(drive=-1)),('drive +1 mV',dict(drive=1))]:
    r=[]
    for s in range(4):
        t,i,n=simulate(seed=s,**{**P,**kw}); r.append(stats(t,n,90))
    r=np.array(r); print('%-15s -> %.1f, %.1f, %.2f, %.1f, %.3f  (bursts sd %.1f)'%(name,*r.mean(0),r[:,1].std())); sys.stdout.flush()
