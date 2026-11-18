import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')
df=pd.read_csv('arms_wells.csv'); n0=len(df); df=df[df.bursts>=3].reset_index(drop=True)
print('SIMULATED wells: %d kept of %d (>=3 bursts). dropped by arm:'%(len(df),n0), (pd.read_csv('arms_wells.csv').groupby('arm').bursts.apply(lambda s:(s<3).sum())).to_dict())
sub=df[df.arm.isin(['rescue','supp_gee','supp_drive'])].reset_index(drop=True); y=(sub.arm!='rescue').astype(int).values
sets={'rate only':['rate'],'aggregate (rate,bursts,interval,frac)':['rate','bursts','interval','frac'],'shape (peak,rise,asym)':['peak','rise','asym'],'all':['rate','bursts','interval','frac','peak','rise','asym']}
models={'logistic':lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),'random forest':lambda:RandomForestClassifier(300,min_samples_leaf=3,random_state=0)}
print('\nRescue vs rate-matched suppression (both mechanisms pooled), each plate held out: mean AUC (worst plate)')
for mn,fn in models.items():
    for sn,f in sets.items():
        a=[]
        for p in sub.plate.unique():
            tr,te=(sub.plate!=p).values,(sub.plate==p).values
            if len(set(y[te]))<2: continue
            m=fn().fit(sub.loc[tr,f],y[tr]); a.append(roc_auc_score(y[te],m.predict_proba(sub.loc[te,f])[:,1]))
        print('  %-13s %-40s %.3f (%.3f)'%(mn,sn,np.mean(a),np.min(a)))
print('\nMixed-effects: feature ~ arm + (1|plate), reference = rescue')
for f in ['rate','bursts','interval','frac','peak','rise','asym']:
    r=smf.mixedlm(f'{f} ~ C(arm, Treatment("rescue"))',df,groups=df['plate']).fit(reml=True)
    s='  %-8s'%f
    for t in r.params.index:
        if 'arm' in t: s+=' %s %+.2f (p=%.2g)'%(t.split('T.')[-1].rstrip(']'),r.params[t],r.pvalues[t])
    print(s)
