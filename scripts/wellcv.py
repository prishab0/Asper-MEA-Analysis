"""
wellcv.py  -  well-level validation for class decoding.
Usage:
  python wellcv.py wells.csv --features shape_1,shape_2 --rate-col MFR [--perms 2000]
Input CSV: one row per well (or per recording/window), columns:
  well   : well/culture ID (all rows from one well share it; splits are by well)
  label  : class (compound / condition)
  <numeric feature columns>; --features lists the ones to use (required);
  --rate-col names the firing-rate column for a rate-only comparison.
  Do NOT include plate position (e.g. a well index) or plate/day identifiers as features:
  in a layout where treatment sits in fixed positions they encode the label.
Outputs: leave-one-well-out balanced accuracy, true chance 1/k, a well-level
permutation null (reported as its mean AND 95th percentile), p-value,
bootstrap CI over wells, and the same for a rate-only model on identical splits.
"""
import argparse, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--rate-col", default=None)
ap.add_argument("--features", required=True); ap.add_argument("--perms", type=int, default=2000)
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
df = pd.read_csv(a.csv)
assert {"well", "label"} <= set(df.columns), "need columns: well, label"
feats = a.features.split(",")
for c in feats + ([a.rate_col] if a.rate_col else []):
    assert c in df.columns, f"column {c!r} not in file; columns are {list(df.columns)}"
wells = df["well"].astype(str).values
uw = np.unique(wells)
wl = df.groupby(df["well"].astype(str))["label"].first().reindex(uw)
classes = np.unique(wl.values); k = len(classes)
yw = np.array([list(classes).index(l) for l in wl.values])      # label per well
counts = pd.Series(wl.values).value_counts()
print(f"{len(uw)} wells, {k} classes, wells per class: {dict(counts)}")
print(f"True chance for {k} equal classes = {1/k:.3f}.  Effective n = {len(uw)} wells "
      f"(rows from one well are not independent).")
if counts.nunique() > 1: print("WARNING: unequal class sizes; using balanced accuracy.")
widx = np.array([list(uw).index(w) for w in wells])

def loo_wells(X, yw_):
    """leave-one-WELL-out; returns predicted class per well."""
    pred = np.zeros(len(uw), int)
    ylab = yw_[widx]
    for i in range(len(uw)):
        tr = widx != i; te = ~tr
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        m.fit(X[tr], ylab[tr])
        P = m.predict_proba(X[te]); cl = m.classes_
        pred[i] = cl[P.mean(0).argmax()]
    return pred

def evaluate(name, cols):
    X = df[cols].astype(float).values
    pred = loo_wells(X, yw)
    obs = balanced_accuracy_score(yw, pred)
    null = np.array([balanced_accuracy_score(yp := rng.permutation(yw), loo_wells(X, yp))
                     for _ in range(a.perms)])
    p = (1 + (null >= obs).sum()) / (1 + len(null))
    ok = (pred == yw).astype(float)
    boots = []
    for _ in range(a.boot):
        s = rng.integers(0, len(uw), len(uw)); boots.append(ok[s].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n[{name}]  features: {cols}")
    print(f"  observed balanced acc (leave-one-well-out): {obs:.3f}   wells correct: {int(ok.sum())}/{len(uw)}")
    print(f"  95% bootstrap CI over wells (accuracy): [{lo:.3f}, {hi:.3f}]")
    print(f"  permutation null: mean {null.mean():.3f}, 95th pct {np.percentile(null,95):.3f}, "
          f"99th pct {np.percentile(null,99):.3f}   (true chance {1/k:.3f})")
    print(f"  permutation p = {p:.4f}")
    return pred
pa = evaluate("all features", feats)
if a.rate_col:
    pr = evaluate("firing rate only", [a.rate_col])
    print("\nPer-class wells correct (all features | rate only):")
    for j, c in enumerate(classes):
        m = yw == j
        print(f"  {c}: {(pa[m]==yw[m]).sum()}/{m.sum()} | {(pr[m]==yw[m]).sum()}/{m.sum()}")
print("\nReport the null's 95th percentile and the CI, not a single 'chance' number. "
      "Small classes (n=4) make the leave-one-out null fall below 1/k; that is expected.")
