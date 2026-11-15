"""
Power simulation for a chronic-dose rescue study (Angelman model).
Arms per batch: WT vehicle, AS vehicle, AS + unsilencer.  Outcome: decoding score, in points.
Blocked design: every batch contains all arms; analysis = OLS with batch fixed effects.
ALL EFFECT SIZES AND VARIANCES ARE ASSUMPTIONS. No Angelman MEA data was used.
Usage: python ube3a_power.py            (writes ube3a_power.csv and prints tables)
"""
import numpy as np, pandas as pd, itertools
from scipy import stats
rng = np.random.default_rng(1)
GAP = 20.0   # assumed WT minus AS vehicle difference, points
SB = 5.0     # assumed between-batch SD, points
NSIM = 2000

def design(nb, n):
    arm = np.tile(np.repeat([0, 1, 2], n), nb)          # 0 WT, 1 AS-veh, 2 AS-drug
    bat = np.repeat(np.arange(nb), 3 * n)
    return arm, bat

def one(nb, n, sw, rescue, rng):
    arm, bat = design(nb, n)
    mu = np.array([GAP, 0.0, rescue * GAP])[arm]
    y = mu + rng.normal(0, SB, nb)[bat] + rng.normal(0, sw, len(arm))
    # OLS: arm dummies + batch dummies
    X = [np.ones(len(y))]
    X += [(arm == k).astype(float) for k in (0, 2)]            # ref = AS vehicle
    X += [(bat == b).astype(float) for b in range(1, nb)]
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; df = len(y) - X.shape[1]
    cov = (res @ res / df) * np.linalg.inv(X.T @ X)
    p = [2 * stats.t.sf(abs(beta[i] / np.sqrt(cov[i, i])), df) for i in (1, 2)]
    return p[0] < .05, p[1] < .05                               # (WT vs AS gap, drug vs AS vehicle)

rows = []
for nb, n, sw, rescue in itertools.product([1, 2, 3], [3, 4, 6, 8], [6, 10], [0.25, 0.5, 0.75]):
    r = np.array([one(nb, n, sw, rescue, rng) for _ in range(NSIM)])
    rows.append(dict(batches=nb, wells_per_arm_per_batch=n, total_wells_per_arm=nb * n,
                     within_well_SD=sw, rescue_fraction=rescue,
                     power_gap=r[:, 0].mean(), power_rescue=r[:, 1].mean(),
                     power_both=(r[:, 0] & r[:, 1]).mean()))
df = pd.DataFrame(rows); df.to_csv("ube3a_power.csv", index=False)
pd.set_option("display.width", 200)
for sw in (6, 10):
    for resc in (0.5,):
        t = df[(df.within_well_SD == sw) & (df.rescue_fraction == resc)]
        print(f"\nwithin-well SD {sw} pts, rescue {int(resc*100)}% of a {GAP:.0f}-pt gap: power to detect drug vs AS vehicle")
        print(t.pivot(index="wells_per_arm_per_batch", columns="batches", values="power_rescue").round(2))
print("\nSmallest design reaching 80% for BOTH the gap and the rescue (None = not in grid):")
for (sw, resc), g in df.groupby(["within_well_SD", "rescue_fraction"]):
    ok = g[g.power_both >= .8].sort_values(["total_wells_per_arm", "batches"])
    print(f"  SD {sw}, rescue {int(resc*100)}%:", None if ok.empty else
          f"{int(ok.iloc[0].batches)} batch(es) x {int(ok.iloc[0].wells_per_arm_per_batch)} wells/arm = {int(ok.iloc[0].total_wells_per_arm)} wells/arm")
