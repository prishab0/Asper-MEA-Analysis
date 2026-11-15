import sys, json, numpy as np
from net import simulate
def rate(params, seeds=(11,12,13), T=20):
    r=[]
    for s in seeds:
        t,i,n=simulate(*params,T=T,seed=s); r.append(len(t)/T/n)
    return float(np.mean(r))
target=rate((1.5,1.0,0.0)); print('target (rescued) rate %.2f'%target); sys.stdout.flush()
def bisect(f, lo, hi, increasing, n=7):
    for _ in range(n):
        mid=(lo+hi)/2; r=f(mid)
        if (r<target)==increasing: lo=mid
        else: hi=mid
    return (lo+hi)/2
out={'target':target}
out['drive']=bisect(lambda d: rate((1.7,0.6,d)), -6.0, 0.0, True)
print('drive',out['drive']); sys.stdout.flush()
out['gee']=bisect(lambda g: rate((g,0.6,0.0)), 1.0, 1.7, True)
print('gee',out['gee']); sys.stdout.flush()
out['gie']=bisect(lambda g: rate((1.7,g,0.0)), 0.6, 4.0, False)
print('gie',out['gie']); sys.stdout.flush()
for k,p in (('drive',(1.7,0.6,out['drive'])),('gee',(out['gee'],0.6,0.0)),('gie',(1.7,out['gie'],0.0))):
    print(k,'achieved rate %.2f'%rate(p)); sys.stdout.flush()
json.dump(out,open('calib.json','w'))
