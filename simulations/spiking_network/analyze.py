import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R=json.load(open('wells.json'))
arms=['rescue','supp_drive','supp_gee','supp_gie','disease']
by={a:[r for r in R if r['arm']==a] for a in arms}
print('%-11s %5s %8s %6s %8s %8s %8s'%('arm','wells','rate Hz','bursts','peak','FWHM ms','rise ms'))
for a in arms:
    rs=by[a]; ok=[r for r in rs if 'fwhm' in r]
    print('%-11s %5d %8.2f %6.1f %8.1f %8.0f %8.0f'%(a,len(rs),np.mean([r['rate'] for r in rs]),np.mean([r['nb'] for r in rs]),
          np.mean([r['peak'] for r in ok]),np.mean([r['fwhm'] for r in ok]),np.mean([r['rise'] for r in ok])))
def table(a,b,fs):
    X=[];y=[]
    for lab,arm in ((0,a),(1,b)):
        for r in by[arm]:
            if 'fwhm' in r: X.append([r[f] for f in fs]); y.append(lab)
    return np.array(X),np.array(y)
def lowo_auc(X,y):
    sc=np.zeros(len(y))
    for i in range(len(y)):
        m=np.ones(len(y),bool); m[i]=False
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0)).fit(X[m],y[m]); sc[i]=clf.decision_function(X[i:i+1])[0]
    return roc_auc_score(y,sc)
print('\nRescue vs each suppression mechanism, matched mean rate. Leave-one-well-out AUC (0.5 = chance):')
print('%-12s %8s %8s %12s'%('vs','rate','nb','shape(3)'))
for b in ('supp_drive','supp_gee','supp_gie'):
    out=[]
    for fs in (['rate'],['nb'],['peak','fwhm','rise']):
        X,y=table('rescue',b,fs); out.append(lowo_auc(X,y))
    print('%-12s %8.3f %8.3f %12.3f'%(b,*out))
# power: subsample n wells per arm
rng=np.random.default_rng(0)
print('\nPower: shape AUC when only n wells per arm are run (mean over 200 resamples), with chance-level rate AUC')
for b in ('supp_drive','supp_gee','supp_gie'):
    Xs,ys=table('rescue',b,['peak','fwhm','rise']); Xr,yr=table('rescue',b,['rate'])
    line=[]
    for n in (3,4,6,8,12):
        a_s=[];a_r=[]
        for _ in range(100):
            idx=np.concatenate([rng.choice(np.where(ys==0)[0],n,replace=False),rng.choice(np.where(ys==1)[0],n,replace=False)])
            for X,y,store in ((Xs,ys,a_s),(Xr,yr,a_r)):
                try: store.append(lowo_auc(X[idx],y[idx]))
                except Exception: pass
        line.append('n=%d: shape %.2f rate %.2f'%(n,np.mean(a_s),np.mean(a_r)))
    print(b,' | '.join(line))
