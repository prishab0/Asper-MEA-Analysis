import sys, numpy as np
from net3 import simulate
from tune3 import stats
P=dict(base=5,U=0.02,k=0.8,tau_d=6000,tau_s=150,T=60)
def run(**kw):
    r=np.array([stats(*(lambda o:(o[0],o[2],60))(simulate(seed=s,**{**P,**kw}))) for s in range(3)]); return r.mean(0)
print('condition -> rate, bursts, width, interval, frac (3 seeds, 60 s)')
for name,kw in [('control',{}),('disease g_ie .5',dict(g_ie=.5)),('disease+g_ee .8',dict(g_ie=.5,g_ee=.8)),('disease+g_ee .7',dict(g_ie=.5,g_ee=.7)),('disease+g_ee .6',dict(g_ie=.5,g_ee=.6)),
                ('disease+drive -.5',dict(g_ie=.5,drive=-.5)),('disease+drive -1',dict(g_ie=.5,drive=-1.0)),('disease+drive -1.5',dict(g_ie=.5,drive=-1.5))]:
    print('%-20s -> %.1f, %.1f, %.2f, %.1f, %.2f'%(name,*run(**kw))); sys.stdout.flush()
