#!/usr/bin/env python3
"""Culture-level summary of the evoked decoder comparison.

The 24 recordings come from 14 cultures (name pattern BATCH-DISH-DIV), so the
independent sample is 14. This script parses the COMPARE table printed by
evoked_decoder.py, averages recordings within each culture, and bootstraps over
cultures (10,000 resamples, seed 0).

    python evoked_decoder.py compare > decoder_compare.txt
    python culture_level.py decoder_compare.txt
"""
import re, sys, collections
import numpy as np

rows = collections.defaultdict(list)
pat = re.compile(r'(\d+-\d+)-(\d+)\s+[\d.]+%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%')
for line in open(sys.argv[1]):
    m = pat.search(line)
    if m:
        rows[m.group(1)].append([float(m.group(i)) for i in (3, 4, 5)])
if not rows:
    sys.exit('No recording rows found in the input file.')
rec = np.array([r for v in rows.values() for r in v])
C = np.array([np.mean(v, axis=0) for v in rows.values()])
rng = np.random.default_rng(0)
bs = np.array([C[rng.integers(0, len(C), len(C))].mean(axis=0) for _ in range(10000)])
print(f'{len(rec)} recordings, {len(C)} cultures')
print('recording mean  rate %.1f  network %.1f  spatial %.1f' % tuple(rec.mean(axis=0)))
print('culture mean    rate %.1f  network %.1f  spatial %.1f' % tuple(C.mean(axis=0)))
for i, n in enumerate(['rate', 'network', 'spatial']):
    lo, hi = np.percentile(bs[:, i], [2.5, 97.5])
    print(f'  {n:8s} 95% interval over cultures [{lo:.1f}, {hi:.1f}]')
m = C[:, 2] - C[:, 1]
print(f'spatial beats network in {(m > 0).sum()} of {len(C)} cultures, smallest margin {m.min():.1f} points')
