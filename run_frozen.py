"""Frozen-embedding check for the bicuculline plates (day 18 -> day 17), plus a position check.

Run inside ~/burst-shape:   uv run python -u run_frozen.py
Needs the day 17 and day 18 files already downloaded by run_day17.py.

Design (all choices fixed before looking at day 17 output):
  1. Burst embedding is fitted on day 18 bursts only.
  2. The burst-shape coordinate is chosen on day 18 only: the coordinate (1..4) whose per-well mean
     has the highest AUC for bicuculline against control. Ties go to the lower index.
  3. Day 17 bursts are projected into the day 18 embedding by a distance-weighted average of the
     coordinates of their 15 nearest day 18 bursts (1-D Wasserstein distance between shapes).
  4. The day 17 test uses the coordinate and the sign fixed in step 2. No refitting on day 17.
"""
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from burst_shape.folders import get_data_inhibblock_folder
from burst_shape.preprocess import burst_extraction
from burst_shape.spectral_clustering import SpectralClusteringModified, compute_affinity_matrix

BIC = ["A2", "A3", "A4", "A5", "A6", "B2", "B3", "B4", "B5", "B6"]
CONTROL = ["A1", "B1", "C1", "D1", "C2", "C3", "C4", "C5", "C6", "D2", "D3", "D4", "D5", "D6"]
EDGE_CONTROLS = ["A1", "B1"]          # controls that sit among the bicuculline wells
FILES = {
    17: ("ctx_14.03.22_Hertie", "day17_potassium4.2_spikes.csv"),
    18: ("ctx_03.04.22_Hertie", "day18_potassium4.2_spikes2.csv"),
}
K_NN = 15
N_PERM = 5000
rng = np.random.default_rng(0)
root = get_data_inhibblock_folder()


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
bm = np.nan_to_num(np.asarray(bm, dtype=float))
print(f"{bm.shape[0]} bursts, shape matrix {bm.shape}")


def level(name):
    if name in df_bursts.index.names:
        return np.asarray(df_bursts.index.get_level_values(name))
    if name in df_bursts.columns:
        return np.asarray(df_bursts[name])
    if name == "well":   # look the well up from the culture table through the shared index keys
        keys = [k for k in ("drug_label", "div", "well_idx") if k in df_bursts.index.names]
        lut = df_cult["well"] if "well" in df_cult.columns else df_cult.reset_index().set_index(keys)["well"]
        bi = df_bursts.reset_index() if keys[0] in df_bursts.index.names else df_bursts
        return np.asarray([lut.loc[tuple(r)] if len(keys) > 1 else lut.loc[r[0]] for r in bi[keys].itertuples(index=False, name=None)])
    raise KeyError(name)


div_b = level("div")
m18, m17 = div_b == 18, div_b == 17

# 1. embedding on day 18 only
n_jobs = max(1, (os.cpu_count() or 2) - 1)
bm18 = bm[m18]
nn = max(5, bm18.shape[0] // 100)
aff = compute_affinity_matrix(bm18, n_jobs=n_jobs, metric="wasserstein", n_neighbors=nn)
clus = SpectralClusteringModified(n_components_max=10, affinity="precomputed", n_neighbors=nn,
                                  random_state=0, n_jobs=n_jobs, verbose=False).compute_maps(aff)
maps18 = clus.maps_[:, 1:5]            # coordinates 1..4 (index 0 is the trivial eigenvector)

# 3. project day 17 bursts (1-D Wasserstein = L1 distance between cumulative distributions)
cdf18, cdf17 = np.cumsum(bm18, axis=1), np.cumsum(bm[m17], axis=1)
maps17 = np.zeros((cdf17.shape[0], maps18.shape[1]))
for s in range(0, cdf17.shape[0], 2000):
    d = cdist(cdf17[s:s + 2000], cdf18, metric="cityblock")
    idx = np.argpartition(d, K_NN, axis=1)[:, :K_NN]
    dd = np.take_along_axis(d, idx, axis=1)
    w = 1.0 / (dd + 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    maps17[s:s + 2000] = np.einsum("nk,nkc->nc", w, maps18[idx])

wells_b = level("well")
tab = []
for day, mask, maps in ((18, m18, maps18), (17, m17, maps17)):
    g = pd.DataFrame(maps, columns=[f"c{i + 1}" for i in range(maps.shape[1])])
    g["well"] = wells_b[mask]
    g = g.groupby("well").mean().reset_index()
    g["div"] = day
    tab.append(g)
tab = pd.concat(tab)
tab["label"] = tab["well"].isin(BIC).astype(int)
mfr = df_cult.reset_index()[["well", "div", "times", "gid"]]
mfr["MFR"] = mfr["times"].apply(len) / mfr["times"].apply(max) / mfr["gid"].apply(lambda g: len(np.unique(g)))
tab = tab.merge(mfr[["well", "div", "MFR"]], on=["well", "div"])

# 2. choose the coordinate on day 18 only
t18, t17 = tab[tab.div == 18], tab[tab.div == 17]
print("\nDay 18 AUC by coordinate (bic vs control):")
aucs = {c: roc_auc_score(t18.label, t18[c]) for c in ("c1", "c2", "c3", "c4")}
for c, a in aucs.items():
    print(f"  {c}: {a:.3f}")
best = max(aucs, key=lambda c: abs(aucs[c] - 0.5))
sign = 1 if aucs[best] >= 0.5 else -1
print(f"chosen on day 18: {best}, sign {sign:+d}")

# 4. test on day 17 with no refitting
print("\n===== DAY 17 (embedding and coordinate fixed on day 18) =====")
s17 = sign * t17[best]
auc17 = roc_auc_score(t17.label, s17)
u = mannwhitneyu(s17[t17.label == 1], s17[t17.label == 0], alternative="greater")
print(f"{best} AUC on day 17: {auc17:.3f}, one-sided Mann-Whitney p = {u.pvalue:.2g}")
print(f"MFR AUC on day 17: {roc_auc_score(t17.label, t17.MFR):.3f} (day 18: {roc_auc_score(t18.label, t18.MFR):.3f})")
print("all four coordinates on day 17 (for reference, not used for selection):")
for c in ("c1", "c2", "c3", "c4"):
    print(f"  {c}: AUC {roc_auc_score(t17.label, t17[c]):.3f}")

lo = max(tab[tab.label == 0].MFR.min(), tab[tab.label == 1].MFR.min())
hi = min(tab[tab.label == 0].MFR.max(), tab[tab.label == 1].MFR.max())
tr, te = t18[(t18.MFR >= lo) & (t18.MFR <= hi)], t17[(t17.MFR >= lo) & (t17.MFR <= hi)]
print(f"\ncross-plate, pooled rate range {lo:.2f} to {hi:.2f} Hz: train {len(tr)}, test {len(te)} "
      f"({int((te.label == 0).sum())} control, {int((te.label == 1).sum())} bic)")
for name, cols in (("MFR only", ["MFR"]), (f"{best} only", [best])):
    m = make_pipeline(StandardScaler(), LogisticRegression()).fit(tr[cols].values, tr.label.values)
    pred = m.predict(te[cols].values)
    obs = balanced_accuracy_score(te.label.values, pred)
    null = np.array([balanced_accuracy_score(rng.permutation(te.label.values), pred) for _ in range(N_PERM)])
    print(f"  {name:12s} balanced accuracy {obs:.3f}, permutation p = {(np.sum(null >= obs) + 1) / (N_PERM + 1):.4f}")

# position check: are the two edge controls on the control side?
print("\n===== POSITION CHECK (edge controls A1, B1 sit among the bicuculline wells) =====")
for day, t in ((18, t18), (17, t17)):
    v = sign * t[best]
    bic_min = v[t.label == 1].min()
    far_ctl = v[(t.label == 0) & (~t.well.isin(EDGE_CONTROLS))]
    for w in EDGE_CONTROLS:
        x = float(v[t.well == w].iloc[0])
        print(f"  day {day} {w}: {best} = {x:.5f}; bicuculline minimum {bic_min:.5f}; "
              f"other-control median {np.median(far_ctl):.5f}; below every bicuculline well: {x < bic_min}")

tab.to_csv("frozen_wells.csv", index=False)
print("\nsaved frozen_wells.csv")
