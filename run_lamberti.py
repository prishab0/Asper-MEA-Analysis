"""Paired baseline vs baseline-after-APV analysis on the Lamberti et al. Dryad data.

Run from inside the burst-shape repo:  uv run python run_lamberti.py
Optional:  LAMBERTI_ROOT=~/lamberti  MIN_SBURST=100  uv run python run_lamberti.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr

from burst_shape.preprocess import burst_extraction
from burst_shape.spectral_clustering import (
    SpectralClusteringModified,
    compute_affinity_matrix,
)

ROOT = os.path.expanduser(os.environ.get("LAMBERTI_ROOT", "~/lamberti"))
MIN_SBURST = int(os.environ.get("MIN_SBURST", "100"))
FS = 16000.0
STIM_CHANNEL_MIN = 60  # electrode ids >= this are treated as non-recording markers
MIN_BURSTS = 5  # per recording, for the per-culture permutation test
N_PERM = 2000
RATE_MATCH_FACTOR = 1.5  # "rate not changed" = within 1.5x either way
rng = np.random.default_rng(0)


def load_spikes(path):
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
    t = np.atleast_1d(np.asarray(d.Ts, dtype=float)) / FS
    c = np.atleast_1d(np.asarray(d.Cs)).astype(int)
    keep = c < STIM_CHANNEL_MIN
    t, c = t[keep], c[keep]
    order = np.argsort(t, kind="stable")
    return t[order], c[order]


# ---------------------------------------------------------------- load
rows = []
for f in sorted(glob.glob(ROOT + "/**/*.mat", recursive=True)):
    name = os.path.basename(f)
    m = re.match(r"exp(\d+)-(\d+)uM-(.*)\.mat", name, flags=re.I)
    if not m:
        continue
    exp, dose, cond = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    if cond not in ("baseline", "baselineapv"):
        continue  # post-stimulation recordings are not used in the paired check
    t, c = load_spikes(f)
    dur = t.max() - t.min() if len(t) > 1 else np.nan
    rows.append(
        dict(
            dose=dose, exp=exp, cond="apv" if cond == "baselineapv" else "base",
            times=t - t.min(), gid=c, duration_s=dur, n_spikes=len(t),
            n_channels=len(np.unique(c)),
        )
    )
    print(f"loaded {name}: {len(t)} spikes, {dur:.0f} s, {len(np.unique(c))} channels")

df = pd.DataFrame(rows)
df["MFR"] = df["n_spikes"] / df["duration_s"]  # whole-array spikes/s
df["MFR_per_channel"] = df["MFR"] / df["n_channels"].clip(lower=1)
df["culture"] = df["dose"].astype(str) + "uM_exp" + df["exp"].astype(str)
df["drug_label"] = df["cond"]
df["div"] = df["dose"]
df["well_idx"] = df["exp"]
df = df.set_index(["drug_label", "div", "well_idx"]).sort_index()

# ---------------------------------------------------------------- bursts
params = dict(
    dataset="inhibblock", maxISIstart=20, maxISIb=20, minBdur=50, minIBI=100,
    minSburst=MIN_SBURST, bin_size=None, n_bins=50, extend_left=0, extend_right=0,
    burst_length_threshold=None, pad_right=False, normalization="integral",
    min_length=30,
)
df_cult, df_bursts, bm = burst_extraction.extract_bursts(
    construct_df_cultures=lambda: df.copy(), **params
)
print(f"\n{bm.shape[0]} bursts total (minSburst={MIN_SBURST})")
print(df_cult.reset_index()[["drug_label", "div", "well_idx", "n_bursts"]]
      .pivot_table(index=["div", "well_idx"], columns="drug_label", values="n_bursts")
      .to_string())

# ---------------------------------------------------------------- shape embedding
n_jobs = max(1, (os.cpu_count() or 2) - 1)
nn = max(5, bm.shape[0] // 100)
aff = compute_affinity_matrix(bm, n_jobs=n_jobs, metric="wasserstein", n_neighbors=nn)
clus = SpectralClusteringModified(
    n_components_max=10, affinity="precomputed", n_neighbors=nn, random_state=0,
    n_jobs=n_jobs, verbose=False,
).compute_maps(aff)
emb = clus.maps_[:, 1:3]
df_bursts["shape_1"], df_bursts["shape_2"] = emb[:, 0], emb[:, 1]

# ---------------------------------------------------------------- paired analysis
idx_names = df_cult.index.names
b = df_bursts.reset_index()
res = []
for (dose, exp), g in df.reset_index().groupby(["div", "well_idx"]):
    r = {"dose": dose, "exp": exp}
    pts = {}
    for cond in ("base", "apv"):
        gg = b[(b["drug_label"] == cond) & (b["div"] == dose) & (b["well_idx"] == exp)]
        pts[cond] = gg[["shape_1", "shape_2"]].values
        row = df.loc[(cond, dose, exp)]
        r[f"MFR_{cond}"] = row["MFR"]
        r[f"n_bursts_{cond}"] = len(gg)
    r["rate_ratio"] = r["MFR_apv"] / r["MFR_base"]
    a, p = pts["base"], pts["apv"]
    if len(a) >= MIN_BURSTS and len(p) >= MIN_BURSTS:
        obs = np.linalg.norm(a.mean(0) - p.mean(0))
        pooled = np.vstack([a, p])
        null = []
        for _ in range(N_PERM):
            perm = rng.permutation(len(pooled))
            pa, pp = pooled[perm[: len(a)]], pooled[perm[len(a):]]
            null.append(np.linalg.norm(pa.mean(0) - pp.mean(0)))
        r["shape_shift"] = obs
        r["shape_p"] = (np.sum(np.array(null) >= obs) + 1) / (N_PERM + 1)
    else:
        r["shape_shift"], r["shape_p"] = np.nan, np.nan
    res.append(r)
res = pd.DataFrame(res).sort_values(["dose", "exp"])
res["rate_matched"] = (res["rate_ratio"] > 1 / RATE_MATCH_FACTOR) & (
    res["rate_ratio"] < RATE_MATCH_FACTOR
)
res["shape_changed"] = res["shape_p"] < 0.05
pd.set_option("display.width", 200)
print("\n===== PER-CULTURE PAIRED RESULTS =====")
print(res.round(4).to_string(index=False))

ok = res.dropna(subset=["shape_p"])
print(f"\ncultures with enough bursts in both recordings: {len(ok)} / {len(res)}")
print(f"shape changed (permutation p<0.05): {int(ok['shape_changed'].sum())}")
print(f"rate changed more than {RATE_MATCH_FACTOR}x: {int((~ok['rate_matched']).sum())}")
print(f"shape changed AND rate within {RATE_MATCH_FACTOR}x: "
      f"{int((ok['shape_changed'] & ok['rate_matched']).sum())}")
print(f"rate-matched cultures (any shape result): {int(ok['rate_matched'].sum())}")
if len(ok) > 3:
    rho, pr = spearmanr(np.abs(np.log2(ok["rate_ratio"])), ok["shape_shift"])
    print(f"Spearman |log2 rate ratio| vs shape shift: rho={rho:.2f}, p={pr:.3f}")
res.to_csv("lamberti_paired_results.csv", index=False)
print("\nsaved lamberti_paired_results.csv")
