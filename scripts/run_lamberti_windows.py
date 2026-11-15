"""Within-culture rate-matched window check on the Lamberti APV data.
Splits each recording into WIN_S windows, keeps only windows whose firing rate lies in the
range both conditions share, and tests burst shape between baseline and APV inside that band.
Also reports a drift reference: shape shift between first and second half of the baseline.
Run inside ~/burst-shape:  MIN_SBURST=30 uv run python -u run_lamberti_windows.py
"""
import glob, os, re
import numpy as np
import pandas as pd
import scipy.io as sio
from burst_shape.preprocess import burst_extraction
from burst_shape.spectral_clustering import SpectralClusteringModified, compute_affinity_matrix

ROOT = os.path.expanduser(os.environ.get("LAMBERTI_ROOT", "~/lamberti"))
MIN_SBURST = int(os.environ.get("MIN_SBURST", "30"))
WIN_S = float(os.environ.get("WIN_S", "60"))
FS = 16000.0
MIN_WIN = 3          # matched windows needed per condition
MIN_BURSTS = 5       # bursts needed per condition inside the band
N_PERM = 2000
rng = np.random.default_rng(0)

rows = []
for f in sorted(glob.glob(ROOT + "/**/*.mat", recursive=True)):
    m = re.match(r"exp(\d+)-(\d+)uM-(.*)\.mat", os.path.basename(f), flags=re.I)
    if not m or m.group(3).lower() not in ("baseline", "baselineapv"):
        continue
    d = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["data"]
    t = np.atleast_1d(np.asarray(d.Ts, dtype=float)) / FS
    c = np.atleast_1d(np.asarray(d.Cs)).astype(int)
    k = c < 60
    t, c = t[k], c[k]
    o = np.argsort(t, kind="stable")
    t, c = t[o], c[o]
    t0 = t.min()
    rows.append(dict(dose=int(m.group(2)), exp=int(m.group(1)),
                     cond="apv" if m.group(3).lower() == "baselineapv" else "base",
                     times=t - t0, gid=c, duration_s=t.max() - t0))
df = pd.DataFrame(rows)
df["drug_label"], df["div"], df["well_idx"] = df["cond"], df["dose"], df["exp"]
df = df.set_index(["drug_label", "div", "well_idx"]).sort_index()

params = dict(dataset="inhibblock", maxISIstart=20, maxISIb=20, minBdur=50, minIBI=100,
              minSburst=MIN_SBURST, bin_size=None, n_bins=50, extend_left=0, extend_right=0,
              burst_length_threshold=None, pad_right=False, normalization="integral", min_length=30)
df_cult, df_bursts, bm = burst_extraction.extract_bursts(construct_df_cultures=lambda: df.copy(), **params)
n_jobs = max(1, (os.cpu_count() or 2) - 1)
nn = max(5, bm.shape[0] // 100)
aff = compute_affinity_matrix(bm, n_jobs=n_jobs, metric="wasserstein", n_neighbors=nn)
clus = SpectralClusteringModified(n_components_max=10, affinity="precomputed", n_neighbors=nn,
                                  random_state=0, n_jobs=n_jobs, verbose=False).compute_maps(aff)
df_bursts["shape_1"], df_bursts["shape_2"] = clus.maps_[:, 1], clus.maps_[:, 2 - 1 + 0] if False else clus.maps_[:, 2]
b = df_bursts.reset_index()
b["t_s"] = b["start_orig"] / 1000.0


def perm_shift(a, p):
    obs = np.linalg.norm(a.mean(0) - p.mean(0))
    pool = np.vstack([a, p])
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        q = rng.permutation(len(pool))
        null[i] = np.linalg.norm(pool[q[:len(a)]].mean(0) - pool[q[len(a):]].mean(0))
    return obs, (np.sum(null >= obs) + 1) / (N_PERM + 1)


out = []
for (dose, exp), _ in df.reset_index().groupby(["div", "well_idx"]):
    r = {"dose": dose, "exp": exp}
    win = {}
    for cond in ("base", "apv"):
        rec = df.loc[(cond, dose, exp)]
        edges = np.arange(0, rec["duration_s"] + WIN_S, WIN_S)
        edges = edges[edges <= rec["duration_s"] + 1e-9]
        rate = np.histogram(rec["times"], bins=edges)[0] / WIN_S
        win[cond] = (edges, rate)
    lo = max(win["base"][1].min(), win["apv"][1].min())
    hi = min(win["base"][1].max(), win["apv"][1].max())
    r["band_lo"], r["band_hi"] = lo, hi
    pts = {}
    for cond in ("base", "apv"):
        edges, rate = win[cond]
        ok_w = np.where((rate >= lo) & (rate <= hi))[0]
        r[f"n_win_{cond}"] = len(ok_w)
        bb = b[(b["drug_label"] == cond) & (b["div"] == dose) & (b["well_idx"] == exp)]
        widx = np.clip(np.searchsorted(edges, bb["t_s"].values, side="right") - 1, 0, len(rate) - 1)
        sel = bb[np.isin(widx, ok_w)]
        pts[cond] = sel[["shape_1", "shape_2"]].values
        r[f"n_bursts_{cond}"] = len(sel)
    r["rate_in_band_base"] = np.mean([win["base"][1][(win["base"][1] >= lo) & (win["base"][1] <= hi)]]) if r["n_win_base"] else np.nan
    r["rate_in_band_apv"] = np.mean([win["apv"][1][(win["apv"][1] >= lo) & (win["apv"][1] <= hi)]]) if r["n_win_apv"] else np.nan
    if r["n_win_base"] >= MIN_WIN and r["n_win_apv"] >= MIN_WIN and min(len(pts["base"]), len(pts["apv"])) >= MIN_BURSTS and lo < hi:
        r["matched_shift"], r["matched_p"] = perm_shift(pts["base"], pts["apv"])
    else:
        r["matched_shift"], r["matched_p"] = np.nan, np.nan
    bb = b[(b["drug_label"] == "base") & (b["div"] == dose) & (b["well_idx"] == exp)]
    half = df.loc[("base", dose, exp)]["duration_s"] / 2
    h1, h2 = bb[bb["t_s"] < half][["shape_1", "shape_2"]].values, bb[bb["t_s"] >= half][["shape_1", "shape_2"]].values
    r["drift_shift"] = np.linalg.norm(h1.mean(0) - h2.mean(0)) if min(len(h1), len(h2)) >= MIN_BURSTS else np.nan
    out.append(r)
res = pd.DataFrame(out).sort_values(["dose", "exp"])
res["shift_over_drift"] = res["matched_shift"] / res["drift_shift"]
pd.set_option("display.width", 250)
print("\n===== RATE-MATCHED WINDOW CHECK (window = %.0f s, MIN_SBURST=%d) =====" % (WIN_S, MIN_SBURST))
print(res.round(4).to_string(index=False))
ev = res.dropna(subset=["matched_p"])
print(f"\ncultures with a usable rate-matched band: {len(ev)} / {len(res)}")
print(f"of those, shape differs at matched rate (p<0.05): {int((ev['matched_p'] < 0.05).sum())}")
print(f"of those, matched shift larger than baseline drift shift: {int((ev['shift_over_drift'] > 1).sum())}")
res.to_csv("lamberti_window_results.csv", index=False)
print("saved lamberti_window_results.csv")
