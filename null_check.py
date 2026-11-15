import numpy as np, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
rng=np.random.default_rng(0)
y=np.repeat(np.arange(6),4)
def loo(X,y):
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))
    return accuracy_score(y,cross_val_predict(m,X,y,cv=LeaveOneOut()))
for nf in (2,5,20):
    X=rng.normal(size=(24,nf))             # pure noise, no signal
    # true labels on noise: repeat over datasets
    obs=[loo(rng.normal(size=(24,nf)),y) for _ in range(60)]
    nullp=[loo(X,rng.permutation(y)) for _ in range(300)]
    print(f"features={nf}: noise data, true labels, LOO acc mean={np.mean(obs):.3f} (1/6={1/6:.3f}); permuted-label null mean={np.mean(nullp):.3f}, 95th pct={np.percentile(nullp,95):.3f}")
