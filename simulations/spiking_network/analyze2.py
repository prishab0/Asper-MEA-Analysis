import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R=[r for r in json.load(open('wells.json')) if 'wave' in r]
for r in R:
    w=np.array(r['wave']); base=np.median(w); half=base+(1-base)/2
    p=50; lo=p
    while lo>0 and w[lo]>half: lo-=1
    hi=p
    while hi<99 and w[hi]>half: hi+=1
    r['base']=base; r['hw']=(hi-lo)*5.0; r['hrise']=(p-lo)*5.0
def lowo(X,y):
    sc=np.zeros(len(y))
    for i in range(len(y)):
        m=np.ones(len(y),bool); m[i]=False
        sc[i]=make_pipeline(StandardScaler(),LogisticRegression()).fit(X[m],y[m]).decision_function(X[i:i+1])[0]
    return roc_auc_score(y,sc)
def tab(b,fs):
    X=[];y=[]
    for lab,arm in ((0,'rescue'),(1,b)):
        for r in R:
            if r['arm']==arm: X.append([r[f] for f in fs]); y.append(lab)
    return np.array(X),np.array(y)
print('half-prominence features, mean per arm')
for a in ('rescue','supp_drive','supp_gee','supp_gie','disease'):
    rs=[r for r in R if r['arm']==a]; print('%-11s base %.2f halfwidth %4.0f ms halfrise %3.0f ms'%(a,np.mean([r['base'] for r in rs]),np.mean([r['hw'] for r in rs]),np.mean([r['hrise'] for r in rs])))
print('\nLOWO AUC, rescue vs suppression (24 vs 24 wells)')
print('%-11s %6s %6s %12s %14s'%('vs','rate','nb','shape(old)','shape(revised)'))
for b in ('supp_drive','supp_gee','supp_gie'):
    out=[lowo(*tab(b,f)) for f in (['rate'],['nb'],['peak','fwhm','rise'],['peak','base','hw','hrise'])]
    print('%-11s %6.3f %6.3f %12.3f %14.3f'%(b,*out))
