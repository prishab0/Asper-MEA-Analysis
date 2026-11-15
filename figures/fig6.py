import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8.5})
cult = ["15.1","15.2","15.3","15.4","15.5","20.1","20.2","20.3","20.4","20.5","25.1","25.2","25.3","25.4","25.5"]
ratio = [.1123,.2037,.1186,.4849,.1387,.2980,.4751,.1827,.4168,.2581,.2046,.3474,.4999,.1562,.1361]
# order of ratio above corrected below to match cult order
ratio = [.1123,.2037,.1186,.4849,.1387,.2980,.4751,.1827,.4168,.2581,.2046,.3474,.4999,.1562,.1361]
shift = [.0004,np.nan,.0010,.0005,.0010,.0015,.0012,.0002,.0021,.0011,.0013,np.nan,np.nan,.0020,.0007]
drift = [.0005,.0001,.0004,.0005,.0008,.0003,.0001,.0001,.0001,.0001,.0004,.0002,.0002,.0005,.0000]
x = np.arange(len(cult))
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.25]}, dpi=200)
a.axhspan(1/1.5, 1.5, color="#e4e4de"); a.axhline(1, color="#999", lw=.6)
a.scatter(x, ratio, color="k", s=14, zorder=3)
a.set_yscale("log"); a.set_ylim(.07, 2); a.set_xticks(x[::2]); a.set_xticklabels(cult[::2], rotation=60)
a.set_ylabel("Firing rate, APV / baseline"); a.set_xlabel("dose (uM) and culture number")
a.text(7, 1.05, "matched-rate zone", ha="center", va="bottom", color="#555", fontsize=7.5)
a.set_title("a. No culture is rate-matched", fontsize=9, loc="left")
b.vlines(x, np.array(drift)*1e3, np.nan_to_num(shift, nan=0)*1e3, color="#bbb", lw=.8, where=None) if False else None
for i in x:
    if not np.isnan(shift[i]): b.plot([i, i], [drift[i]*1e3, shift[i]*1e3], color="#bbb", lw=.9, zorder=1)
b.scatter(x, np.array(drift)*1e3, s=16, facecolor="white", edgecolor="#555", zorder=3, label="baseline, first vs second half (drift)")
b.scatter(x, np.array(shift)*1e3, s=16, color="k", zorder=3, label="baseline vs APV")
b.set_xticks(x); b.set_xticklabels(cult, rotation=60, fontsize=7.5)
b.set_ylabel("Burst-shape shift (x 10$^{-3}$)"); b.set_xlabel("dose (uM) and culture number")
b.set_ylim(-0.1, 2.9); b.legend(frameon=False, fontsize=7.2, loc="upper left")
b.set_title("b. Shape shift against drift reference", fontsize=9, loc="left")
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig6.png")
