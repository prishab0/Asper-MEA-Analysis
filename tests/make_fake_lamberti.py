import os, sys
import numpy as np
import scipy.io as sio
from types import SimpleNamespace

root = sys.argv[1]
rng = np.random.default_rng(1)
FS = 16000
for dose in (15, 20, 25):
    for exp in range(1, 4):
        for cond, rate_scale, width in (("baseline", 1.0, 0.15), ("baselineAPV", 0.6, 0.35)):
            d = os.path.join(root, f"{dose}uM", cond)
            os.makedirs(d, exist_ok=True)
            T = 1200.0
            ts, cs = [], []
            for b in np.arange(5, T, 8 + rng.normal(0, 1)):
                n = int(300 * rate_scale * rng.uniform(0.3, 1.8))
                tt = b + np.abs(rng.normal(0, width, n)) * 1.0
                ts.append(tt); cs.append(rng.integers(0, 59, n))
            ts.append(rng.uniform(0, T, 300)); cs.append(rng.integers(0, 59, 300))
            ts = np.concatenate(ts); cs = np.concatenate(cs)
            stim = np.array([100.0, 200.0]); ts = np.concatenate([ts, stim]); cs = np.concatenate([cs, [60, 60]])
            o = np.argsort(ts)
            data = {"Ts": (ts[o] * FS).astype(np.int64), "Cs": cs[o].astype(np.int32),
                    "sampleRate": FS, "numSpikes": len(ts)}
            sio.savemat(os.path.join(d, f"exp{exp}-{dose}uM-{cond}.mat"), {"data": data})
