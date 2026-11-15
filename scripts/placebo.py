import warnings, numpy as np, pandas as pd, re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
df = pd.read_csv("day17_day18_wells.csv")
ctl = df[df["label"] == 0].copy()          # controls only; nothing was applied
col = ctl["well"].str.extract(r"(\d+)")[0].astype(int)
row = ctl["well"].str[0]
tests = {"columns 1-3 vs 4-6": (col >= 4).astype(int),
         "rows A/B vs C/D":    row.isin(["A", "B"]).astype(int)}
uw = np.array(sorted(ctl["well"].unique()))
widx = np.array([list(uw).index(w) for w in ctl["well"]])
X = ctl[["shape_1", "shape_2"]].astype(float).values
def loo(y):
    pred = np.zeros(len(uw), int)
    for i in range(len(uw)):
        tr = widx != i
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
        pred[i] = m.predict(X[~tr]).mean() > 0.5
    return pred
for name, y in tests.items():
    yw = np.array([y[ctl["well"].values == w].iloc[0] for w in uw])
    yr = y.values
    obs = balanced_accuracy_score(yw, loo(yr))
    null = [balanced_accuracy_score(yp := rng.permutation(yw), loo(yp[widx])) for _ in range(500)]
    print(f"{name}: n={len(uw)} control wells ({yw.sum()} vs {len(yw)-yw.sum()}), "
          f"balanced acc {obs:.3f}, null 95th pct {np.percentile(null,95):.3f}, "
          f"p={(1+(np.array(null)>=obs).sum())/501:.3f}")
