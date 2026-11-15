"""Replicate the bicuculline rate-overlap check on a second plate (day 17).

Run inside ~/burst-shape:   uv run python -u run_day17.py
Needs about 2 GB of free disk for the day 17 file (1.7 GB) the first time it runs.
Bursts from day 17 and day 18 are embedded together, so shape coordinates are comparable.
"""
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from burst_shape.folders import get_data_inhibblock_folder
from burst_shape.load import download_inhibblock_data
from burst_shape.preprocess import burst_extraction
from burst_shape.spectral_clustering import SpectralClusteringModified, compute_affinity_matrix

BIC = ["A2", "A3", "A4", "A5", "A6", "B2", "B3", "B4", "B5", "B6"]
CONTROL = ["A1", "B1", "C1", "D1", "C2", "C3", "C4", "C5", "C6", "D2", "D3", "D4", "D5", "D6"]
FILES = {
    17: ("ctx_14.03.22_Hertie", "day17_potassium4.2_spikes.csv"),
    18: ("ctx_03.04.22_Hertie", "day18_potassium4.2_spikes2.csv"),
}
N_PERM = 1000
rng = np.random.default_rng(0)

root = get_data_inhibblock_folder()
for day, (folder, fname) in FILES.items():
    path = os.path.join(root, folder, "extracted_data", fname)
    if os.path.exists(path):
        continue
    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
    need = 2.2 if day == 17 else 0.5
    if free_gb < need:
        raise SystemExit(f"Need about {need} GB free for day {day}, only {free_gb:.1f} GB available. Free some space and rerun.")
    download_inhibblock_data.download(days=[day])


def load_day(day):
    folder, fname = FILES[day]
    path = os.path.join(root, folder, "extracted_data", fname)
    acc = {w: ([], []) for w in BIC + CONTROL}
    for chunk in pd.read_csv(path, usecols=["Timestamp [µs]", "Channel ID", "Well Label"], chunksize=4_000_000):
        chunk = chunk[chunk["Well Label"].isin(acc)]
        for w, g in chunk.groupby("Well Label"):
            acc[w][0].append(g["Timestamp [µs]"].values * 1e-6)
            acc[w][1].append(g["Channel ID"].values)
    rows = []
    for w in BIC + CONTROL:
        t = np.concatenate(acc[w][0]) if acc[w][0] else np.array([])
        c = np.concatenate(acc[w][1]) if acc[w][1] else np.array([])
        o = np.argsort(t, kind="stable")
        rows.append(dict(well=w, div=day, drug_label="bic" if w in BIC else "control", times=t[o], gid=c[o]))
    return pd.DataFrame(rows)


df = pd.concat([load_day(18), load_day(17)]).reset_index(drop=True)
df["well_idx"] = df.groupby(["drug_label", "div"]).cumcount()
df = df.set_index(["drug_label", "div", "well_idx"])

params = dict(dataset="inhibblock", maxISIstart=20, maxISIb=20, minBdur=50, minIBI=100, minSburst=100,
              bin_size=None, n_bins=50, extend_left=0, extend_right=0, burst_length_threshold=None,
              pad_right=False, normalization="integral", min_length=30)
df_cult, df_bursts, bm = burst_extraction.extract_bursts(construct_df_cultures=lambda: df.copy(), **params)
print(f"\n{bm.shape[0]} bursts across both days")

n_jobs = max(1, (os.cpu_count() or 2) - 1)
nn = max(5, bm.shape[0] // 100)
aff = compute_affinity_matrix(bm, n_jobs=n_jobs, metric="wasserstein", n_neighbors=nn)
clus = SpectralClusteringModified(n_components_max=10, affinity="precomputed", n_neighbors=nn,
                                  random_state=0, n_jobs=n_jobs, verbose=False).compute_maps(aff)
df_bursts["shape_1"], df_bursts["shape_2"] = clus.maps_[:, 1], clus.maps_[:, 2]

idx = df_cult.index.names
shape = df_bursts.groupby(idx)[["shape_1", "shape_2"]].mean()
cult = df_cult.join(shape)
cult["MFR"] = cult["times"].apply(len) / cult["times"].apply(max) / cult["gid"].apply(lambda g: len(np.unique(g)))
cult = cult[cult["n_bursts"] > 0].reset_index()
cult["label"] = (cult["drug_label"] == "bic").astype(int)


def loo_perm(X, y, name):
    model = make_pipeline(StandardScaler(), LogisticRegression())
    obs = balanced_accuracy_score(y, cross_val_predict(model, X, y, cv=LeaveOneOut()))
    null = np.array([balanced_accuracy_score(p, cross_val_predict(model, X, p, cv=LeaveOneOut()))
                     for p in (rng.permutation(y) for _ in range(N_PERM))])
    p = (np.sum(null >= obs) + 1) / (N_PERM + 1)
    print(f"  {name:10s} balanced accuracy {obs:.3f}, permutation p = {p:.4f}, null 95th pct {np.percentile(null, 95):.3f}")


def overlap(d):
    lo = max(d[d.label == 0].MFR.min(), d[d.label == 1].MFR.min())
    hi = min(d[d.label == 0].MFR.max(), d[d.label == 1].MFR.max())
    return d[(d.MFR >= lo) & (d.MFR <= hi)], lo, hi


print("\n===== DAY 17 ALONE =====")
d17 = cult[cult["div"] == 17]
print(d17.groupby("drug_label").agg(wells=("MFR", "size"), mean_MFR=("MFR", "mean")).round(3).to_string())
o17, lo, hi = overlap(d17)
print(f"rate-overlap range {lo:.3f} to {hi:.3f}: {int((o17.label == 0).sum())} control, {int((o17.label == 1).sum())} bic")
if min((o17.label == 0).sum(), (o17.label == 1).sum()) >= 3:
    loo_perm(o17[["MFR"]].values, o17.label.values, "MFR only")
    loo_perm(o17[["shape_1", "shape_2"]].values, o17.label.values, "shape_2D")
else:
    print("  too few wells in the overlap range for a test")

print("\n===== CROSS-PLATE: fit on day 18, test on day 17 (wells in the pooled overlap range) =====")
allo, lo2, hi2 = overlap(cult)
tr, te = allo[allo["div"] == 18], allo[allo["div"] == 17]
print(f"pooled overlap range {lo2:.3f} to {hi2:.3f}; train {len(tr)} wells, test {len(te)} wells "
      f"({int((te.label == 0).sum())} control, {int((te.label == 1).sum())} bic)")
if tr.label.nunique() == 2 and te.label.nunique() == 2:
    for name, cols in (("MFR only", ["MFR"]), ("shape_2D", ["shape_1", "shape_2"])):
        m = make_pipeline(StandardScaler(), LogisticRegression()).fit(tr[cols].values, tr.label.values)
        pred = m.predict(te[cols].values)
        obs = balanced_accuracy_score(te.label.values, pred)
        null = np.array([balanced_accuracy_score(rng.permutation(te.label.values), pred) for _ in range(5000)])
        print(f"  {name:10s} balanced accuracy {obs:.3f}, permutation p = {(np.sum(null >= obs) + 1) / 5001:.4f}")
else:
    print("  one of the plates has a single class in the overlap range; no test")

print("\n===== PER-WELL TABLE =====")
print(cult.sort_values(["div", "MFR"])[["div", "well", "drug_label", "MFR", "shape_1", "shape_2"]].round(5).to_string(index=False))
cult.drop(columns=["times", "gid"], errors="ignore").to_csv("day17_day18_wells.csv", index=False)
print("\nsaved day17_day18_wells.csv")
