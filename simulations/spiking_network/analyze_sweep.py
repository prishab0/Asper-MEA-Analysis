import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R=json.load(open('sweep.json'))
F4=['peak','base','hw','hrise']
def feats(rs,cols):
    allv={c:np.median([r[c] for r in R if c in r]) for c in F4}
    X=[]
    for r in rs: X.append([r.get(c,allv.get(c,0)) for c in cols])
    return np.array(X)
def lowo(X,y):
    sc=np.zeros(len(y))
    for i in range(len(y)):
        m=np.ones(len(y),bool); m[i]=False
        sc[i]=make_pipeline(StandardScaler(),LogisticRegression()).fit(X[m],y[m]).decision_function(X[i:i+1])[0]
    return roc_auc_score(y,sc)
print('Leave-one-well-out AUC, rescue vs rate-matched suppression, 12 wells per arm. 0.5 = chance.')
print('jitter x   suppression   rate   burst-count   shape(4)   shape+count   wells w/o bursts')
for s in sorted({r['scale'] for r in R}):
    for b in ('supp_drive','supp_gee','supp_gie'):
        rs=[r for r in R if r['scale']==s and r['arm'] in ('rescue',b)]
        y=np.array([0 if r['arm']=='rescue' else 1 for r in rs])
        for r in rs: r['nbf']=r['nb']
        auc=[lowo(feats(rs,['rate']) if False else np.array([[r['rate']] for r in rs]),y),
             lowo(np.array([[r['nb']] for r in rs]),y),
             lowo(feats(rs,F4),y),
             lowo(np.hstack([feats(rs,F4),np.array([[r['nb']] for r in rs])]),y)]
        miss=sum('hw' not in r for r in rs)
        print('%8.0f   %-11s %6.2f %10.2f %11.2f %12.2f %12d'%(s,b,*auc,miss))
print('\nMean rate by arm (Hz) and CV across wells, per jitter level')
for s in sorted({r['scale'] for r in R}):
    line=[]
    for a in ('rescue','supp_drive','supp_gee','supp_gie'):
        v=np.array([r['rate'] for r in R if r['scale']==s and r['arm']==a]); line.append('%s %.1f (CV %.2f)'%(a,v.mean(),v.std()/v.mean()))
    print('x%-2.0f  '%s+'  '.join(line))
