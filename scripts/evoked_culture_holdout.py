"""Culture-level leakage check for the evoked spatiotemporal decoder.

The headline result in evoked_decoder.py (SPATIAL 36.5%, RATE 4.8%, chance 1.8%)
trains and tests inside one recording. Two questions that design leaves open:

  1. INDEPENDENCE. Recordings come from cultures, and several recordings can
     come from the same culture on different days. If so, "24 of 24 recordings"
     is fewer than 24 independent samples. Mode `inventory` counts cultures and
     `summary` reports the result with culture as the unit, with a bootstrap
     interval over cultures.

  2. GENERALISATION. Mode `transfer` trains on one recording and tests on a
     different one, split by whether the test recording is the same culture on
     another day or a different culture. Stimulus labels are electrode indices
     on a shared array, so a decoder trained on one recording can be scored on
     another. A large drop from within-recording to cross-culture accuracy
     means the 36.5% is culture-specific and not a general property.

Culture id is the first two dash-separated fields of the file name (batch-culture),
for example 1-1-20-... is batch 1, culture 1, day 20. Check the printed inventory
against the dataset documentation before trusting the grouping.

Run from the data directory, as with evoked_decoder.py:

    python /path/to/scripts/evoked_culture_holdout.py inventory
    python /path/to/scripts/evoked_culture_holdout.py summary
    python /path/to/scripts/evoked_culture_holdout.py transfer

Caveats: the labels are shared stimulation sites, but each culture wires those
sites to different responses, so low cross-culture accuracy is expected for
physical reasons and is not by itself a flaw. What matters is the within-culture
figure with culture as the unit. Rat cortical cultures, no compound, nothing
here speaks to rescue versus suppression.
"""
import os
import re
import sys
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evoked_decoder as ed  # noqa: E402


def culture_of(stem):
    name = os.path.basename(stem)
    m = re.match(r'^(\d+)-(\d+)-(\d+)', name)
    if m:
        return '%s-%s' % (m.group(1), m.group(2)), int(m.group(3))
    return os.path.basename(os.path.dirname(stem)) or name, -1


def collect(n_rec):
    """Features and labels per usable recording."""
    out = []
    for stem in ed.recordings()[:n_rec]:
        try:
            rate, network, spatial, y, chance, _ = ed.load_features(stem)
        except Exception:
            continue
        cult, div = culture_of(stem)
        out.append(dict(stem=stem, culture=cult, div=div, rate=rate,
                        network=network, spatial=spatial, y=y, chance=chance))
    return out


def run_inventory(n_rec):
    recs = collect(n_rec)
    if not recs:
        print('No usable recordings. Are you in the download directory?')
        return
    by_c = defaultdict(list)
    for r in recs:
        by_c[r['culture']].append(r)
    print('%-12s %5s  %s' % ('CULTURE', 'RECS', 'DAYS'))
    print('-' * 50)
    for c, rs in sorted(by_c.items()):
        print('%-12s %5d  %s' % (c, len(rs), sorted(r['div'] for r in rs)))
    print('-' * 50)
    print('%d recordings from %d cultures' % (len(recs), len(by_c)))
    if len(by_c) < len(recs):
        print('Recordings are NOT independent: effective n is cultures, not recordings.')


def run_summary(n_rec, n_boot=10000):
    recs = collect(n_rec)
    if not recs:
        print('No usable recordings. Are you in the download directory?')
        return
    rows = []
    for r in recs:
        rows.append((r['culture'],
                     ed.cv_score(r['rate'], r['y']),
                     ed.cv_score(r['network'], r['y']),
                     ed.cv_score(r['spatial'], r['y']),
                     r['chance']))
    by_c = defaultdict(list)
    for row in rows:
        by_c[row[0]].append(row[1:])
    cult = np.array([np.mean(v, axis=0) for v in by_c.values()]) * 100
    allrec = np.array([row[1:] for row in rows]) * 100

    print('%-14s %8s %8s %8s %8s' % ('UNIT', 'CHANCE', 'RATE', 'NETWORK', 'SPATIAL'))
    print('-' * 52)
    print('%-14s %7.1f%% %7.1f%% %7.1f%% %7.1f%%'
          % ('recordings (%d)' % len(allrec), allrec[:, 3].mean(),
             allrec[:, 0].mean(), allrec[:, 1].mean(), allrec[:, 2].mean()))
    print('%-14s %7.1f%% %7.1f%% %7.1f%% %7.1f%%'
          % ('cultures (%d)' % len(cult), cult[:, 3].mean(),
             cult[:, 0].mean(), cult[:, 1].mean(), cult[:, 2].mean()))

    rng = np.random.default_rng(ed.SEED)
    print('\nBootstrap over cultures (%d resamples), 95%% interval:' % n_boot)
    for j, name in enumerate(['RATE', 'NETWORK', 'SPATIAL']):
        boots = [cult[rng.integers(0, len(cult), len(cult)), j].mean()
                 for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print('  %-8s %5.1f%%  [%5.1f, %5.1f]' % (name, cult[:, j].mean(), lo, hi))
    gap = cult[:, 2] - cult[:, 0]
    wins = int((gap > 0).sum())
    print('\nSPATIAL beat RATE in %d of %d cultures (min margin %.1f points).'
          % (wins, len(cult), gap.min()))
    print('Report the culture-level interval, not the recording-level mean.')


def run_transfer(n_rec):
    """Train on one recording, test on another, grouped by relationship."""
    recs = collect(n_rec)
    if len(recs) < 2:
        print('Not enough recordings.')
        return
    same_c, diff_c, within = [], [], []
    for i, a in enumerate(recs):
        within.append(ed.cv_score(a['spatial'], a['y'], n_trees=150))
        for j, b in enumerate(recs):
            if i == j or a['spatial'].shape[1] != b['spatial'].shape[1]:
                continue
            seen = np.isin(b['y'], np.unique(a['y']))
            if seen.sum() < 50:
                continue
            model = ed.classifier(150).fit(a['spatial'], a['y'])
            acc = (model.predict(b['spatial'][seen]) == b['y'][seen]).mean()
            (same_c if a['culture'] == b['culture'] else diff_c).append(acc)
    chance = float(np.mean([r['chance'] for r in recs]))
    print('%-34s %6s %8s' % ('TRAIN -> TEST', 'PAIRS', 'ACC'))
    print('-' * 52)
    print('%-34s %6d %7.1f%%' % ('same recording (shuffled CV)', len(within), 100 * np.mean(within)))
    if same_c:
        print('%-34s %6d %7.1f%%' % ('same culture, other day', len(same_c), 100 * np.mean(same_c)))
    else:
        print('%-34s %6s %8s' % ('same culture, other day', 0, 'n/a'))
    if diff_c:
        print('%-34s %6d %7.1f%%' % ('DIFFERENT culture', len(diff_c), 100 * np.mean(diff_c)))
    print('%-34s %6s %7.1f%%' % ('chance', '', 100 * chance))
    print('\nCross-culture accuracy near chance means the decoder is culture-specific:')
    print('the result supports per-culture decoding (the assay design), not a')
    print('universal readout. Say that plainly in any write-up.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('analysis', choices=['inventory', 'summary', 'transfer', 'all'])
    ap.add_argument('-n', type=int, default=24)
    args = ap.parse_args()
    jobs = ['inventory', 'summary', 'transfer'] if args.analysis == 'all' else [args.analysis]
    for job in jobs:
        print('\n' + '=' * 60 + '\n' + job.upper() + '\n' + '=' * 60)
        {'inventory': run_inventory, 'summary': run_summary,
         'transfer': run_transfer}[job](args.n)


if __name__ == '__main__':
    main()
