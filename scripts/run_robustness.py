"""Robustness of the bicuculline results to analysis settings, with bootstrap intervals.

Run inside ~/burst-shape:   uv run python -u run_robustness.py
Needs day 17 and day 18 spike files already downloaded (run_day17.py does this).
Takes a while: one joint embedding per minSburst setting.

For each minSburst in MIN_SBURSTS and each number of shape dimensions in DIMS it reports
  day 18 within-plate: leave-one-out logistic regression in the shared rate range (1000 permutations)
  cross-plate: fit on day 18, test on day 17 (5000 permutations)
and 95% bootstrap intervals (wells resampled with replacement, 2000 draws) for each balanced accuracy.
Also repeats the within-plate and cross-plate tests with the shared rate range shrunk to its central 80%.
Writes robustness_results.csv. Seed 0.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from burst_shape.folders import get_data_inhibblock_folder
from burst_shape.preprocess import burst_extraction
from burst_shape.spectral_clustering import SpectralClusteringModified, compute_affinity_matrix

MIN_SBURSTS = [50, 100, 200]
DIMS = [1, 2, 3]            # number of shape coordinates used (shape_1 .. shape_k)
N_PERM_WITHIN, N_PERM_CROSS, N_BOOT = 1000, 5000, 2000
BIC = ["A2", "A3", "A4", "A5", "A6", "B2", "B3", "B4", "B5", "B6"]
CONTROL = ["A1", "B1", "C1", "D1", "C2", "C3", "C4", "C5", "C6", "D2", "D3", "D4", "D5", "D6"]
FILES = {17: ("ctx_14.03.22_Hertie", "day17_potassium4.2_spikes.csv"),
         18: ("ctx_03.04.22_Hertie", "day18_potassium4.2_spikes2.csv")}
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


raw = pd.concat([load_day(18), load_day(17)]).reset_index(drop=True)
raw["well_idx"] = raw.groupby(["drug_label", "div"]).cumcount()
raw = raw.set_index(["drug_label", "div", "well_idx"])


def boot_ci(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(balanced_accuracy_score(y[i], pred[i]))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def within(d, cols):
    model = make_pipeline(StandardScaler(), LogisticRegression())
    pred = cross_val_predict(model, d[cols].values, d.label.values, cv=LeaveOneOut())
    obs = balanced_accuracy_score(d.label.values, pred)
    null = np.array([balanced_accuracy_score(p, cross_val_predict(model, d[cols].values, p, cv=LeaveOneOut()))
                     for p in (rng.permutation(d.label.values) for _ in range(N_PERM_WITHIN))])
    return obs, (np.sum(null >= obs) + 1) / (N_PERM_WITHIN + 1), boot_ci(d.label.values, pred)


def cross(tr, te, cols):
    m = make_pipeline(StandardScaler(), LogisticRegression()).fit(tr[cols].values, tr.label.values)
    pred = m.predict(te[cols].values)
    obs = balanced_accuracy_score(te.label.values, pred)
    null = np.array([balanced_accuracy_score(rng.permutation(te.label.values), pred) for _ in range(N_PERM_CROSS)])
    return obs, (np.sum(null >= obs) + 1) / (N_PERM_CROSS + 1), boot_ci(te.label.values, pred)


def shared_range(d, scale):
    lo = max(d[d.label == 0].MFR.min(), d[d.label == 1].MFR.min())
    hi = min(d[d.label == 0].MFR.max(), d[d.label == 1].MFR.max())
    if scale < 1 and hi > lo:
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * scale
        lo, hi = mid - half, mid + half
    return lo, hi


results = []
for ms in MIN_SBURSTS:
    print(f"\n######## minSburst = {ms} ########")
    params = dict(dataset="inhibblock", maxISIstart=20, maxISIb=20, minBdur=50, minIBI=100, minSburst=ms,
                  bin_size=None, n_bins=50, extend_left=0, extend_right=0, burst_length_threshold=None,
                  pad_right=False, normalization="integral", min_length=30)
    df_cult, df_bursts, bm = burst_extraction.extract_bursts(construct_df_cultures=lambda: raw.copy(), **params)
    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    nn = max(5, bm.shape[0] // 100)
    aff = compute_affinity_matrix(bm, n_jobs=n_jobs, metric="wasserstein", n_neighbors=nn)
    clus = SpectralClusteringModified(n_components_max=10, affinity="precomputed", n_neighbors=nn,
                                      random_state=0, n_jobs=n_jobs, verbose=False).compute_maps(aff)
    for k in range(1, 6):
        df_bursts[f"shape_{k}"] = clus.maps_[:, k]
    idx = df_cult.index.names
    cult = df_cult.join(df_bursts.groupby(idx)[[f"shape_{k}" for k in range(1, 6)]].mean())
    cult["MFR"] = cult["times"].apply(len) / cult["times"].apply(max) / cult["gid"].apply(lambda g: len(np.unique(g)))
    cult = cult[cult["n_bursts"] > 0].reset_index()
    cult["label"] = (cult["drug_label"] == "bic").astype(int)
    print(f"{bm.shape[0]} bursts, {len(cult)} wells with bursts")
    for scale in (1.0, 0.8):
        lo, hi = shared_range(cult, scale)
        sub = cult[(cult.MFR >= lo) & (cult.MFR <= hi)]
        d18, d17 = sub[sub["div"] == 18], sub[sub["div"] == 17]
        for k in DIMS:
            cols = [f"shape_{j}" for j in range(1, k + 1)]
            for name, c in (("MFR only", ["MFR"]), (f"shape_{k}D", cols)):
                if d18.label.nunique() == 2 and min(d18.label.value_counts()) >= 3:
                    o, p, ci = within(d18, c)
                    results.append(dict(minSburst=ms, range_scale=scale, test="day18 within", features=name, n=len(d18),
                                        bal_acc=o, perm_p=p, ci_lo=ci[0], ci_hi=ci[1]))
                if tr_ok := (d18.label.nunique() == 2 and d17.label.nunique() == 2):
                    o, p, ci = cross(d18, d17, c)
                    results.append(dict(minSburst=ms, range_scale=scale, test="train18 test17", features=name, n=len(d17),
                                        bal_acc=o, perm_p=p, ci_lo=ci[0], ci_hi=ci[1]))
        print(f"  range scale {scale}: {len(d18)} day 18 wells, {len(d17)} day 17 wells")

res = pd.DataFrame(results).drop_duplicates(["minSburst", "range_scale", "test", "features"])
pd.set_option("display.width", 200)
print("\n===== SUMMARY =====")
print(res.round(3).to_string(index=False))
res.to_csv("robustness_results.csv", index=False)
print("\nsaved robustness_results.csv")
