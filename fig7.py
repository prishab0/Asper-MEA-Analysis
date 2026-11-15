import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8.5})
t17="""bic 15.41177 0.00090
bic 17.67321 0.00176
bic 18.73990 0.00004
bic 19.91588 0.00126
bic 20.01741 -0.00009
bic 23.33530 -0.00001
bic 24.50114 0.00089
bic 25.29066 0.00122
control 25.54595 -0.00057
control 26.61464 -0.00058
bic 26.74533 0.00089
control 28.66374 -0.00030
bic 29.38511 0.00230
control 30.95051 -0.00055
control 34.66619 -0.00102
control 35.04808 -0.00039
control 38.90378 -0.00020
control 39.71906 -0.00071
control 40.72016 -0.00064
control 42.21262 -0.00021
control 42.50814 -0.00066
control 45.29521 -0.00028
control 46.55074 -0.00074
control 65.27702 -0.00102"""
t18="""control 0.38186 0.00037
control 2.76036 -0.00105
control 3.40545 0.00024
control 3.63470 -0.00128
control 3.82569 -0.00088
bic 4.68140 0.00111
control 4.69462 -0.00148
bic 4.89353 0.00179
bic 5.21412 0.00174
control 6.10104 -0.00047
control 6.57361 0.00016
control 7.58086 -0.00029
control 7.86984 -0.00013
control 8.04517 -0.00080
bic 8.81783 0.00115
control 9.06834 -0.00097
bic 10.30622 0.00113
control 12.00094 0.00009
control 12.55458 -0.00153
bic 13.91146 0.00258
bic 14.40258 0.00083
bic 15.34024 0.00263
bic 17.68970 0.00189
bic 18.15153 0.00265"""
def parse(t): return [(l.split()[0], float(l.split()[1]), float(l.split()[2])*1e3) for l in t.split("\n")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True, dpi=200)
for ax, (name, d, band) in zip(axs, (("a. Plate 1 (day 18)", parse(t18), (4.68, 12.55)), ("b. Plate 2 (day 17)", parse(t17), (25.55, 29.39)))):
    ax.axvspan(*band, color="#e4e4de", zorder=0)
    for lab, x, y in d:
        if lab == "bic": ax.scatter(x, y, s=18, color="k", zorder=3)
        else: ax.scatter(x, y, s=18, facecolor="white", edgecolor="#444", zorder=3)
    ax.axhline(0, color="#aaa", lw=.6, zorder=1)
    ax.set_xscale("log"); ax.set_title(name, fontsize=9, loc="left")
    ax.set_xlabel("Mean firing rate (Hz per channel)")
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("Second burst-shape coordinate (x 10$^{-3}$)")
axs[0].scatter([], [], s=18, color="k", label="bicuculline"); axs[0].scatter([], [], s=18, facecolor="white", edgecolor="#444", label="control")
axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig("fig7.png")
