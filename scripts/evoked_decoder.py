#!/usr/bin/env python3
"""
Aggregate network measures discard stimulus information in evoked MEA recordings.

Reproduces every number in the accompanying note, from public data.

    Wagenaar DA, Pine J, Potter SM. An extremely rich repertoire of bursting
    patterns during the development of cortical cultures.
    BMC Neuroscience 2006;7:11.

Get the data (about 1.5 GB of compact spike and stimulus files):

    wget -r -np -nH --cut-dirs=1 -A "*.tri.txt.bz2,*.spk.txt.bz2" \
        https://potterlab.bme.gatech.edu/development-data/

Then, from the download directory:

    python evoked_decoder.py compare      # the headline comparison
    python evoked_decoder.py controls     # artifact, leakage, shuffle
    python evoked_decoder.py rate         # firing rate vs decoding
    python evoked_decoder.py power        # paired endpoint and power
    python evoked_decoder.py all

Requires numpy and scikit-learn. Seed is fixed at 0 throughout.

The question
------------
For each delivered stimulus, can a classifier recover WHICH electrode was
stimulated from the network response alone? Three representations of the same
trials go through the same classifier, so the only variable is how the response
is described:

    RATE      total spike count                                  1 feature
    NETWORK   aggregate descriptors, no electrode identity      10 features
    SPATIAL   per-electrode counts in four time bins       n_elec * 4

NETWORK is the honest comparator. SPATIAL beating RATE proves nothing: a scalar
cannot name one of N electrodes. The claim only means something if SPATIAL
beats NETWORK.

Caveats, stated up front
------------------------
Rat cortical cultures, not human iPSC. No compound was administered, so nothing
here speaks to separating functional rescue from suppression. Stimulation was
single-electrode in sequence, not designed spatial patterns. The NETWORK feature
set was constructed for this analysis and a published comparator would be
stronger. Cross-culture generalisation is untested.

Prisha Bhanushali, independent researcher. prishabhanus@umass.edu
"""
import sys
import os
import bz2
import glob
import math
import random
import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 0
NBINS = 4             # time bins within the response window
BLANK = 0.020         # s, start of window: excludes artifact and direct activation
WINDOW = 0.080        # s, window length: 20 to 100 ms post-stimulus
MIN_TRIALS = 10       # drop stimulus sites with fewer trials than this


# --------------------------------------------------------------------- io

def load(path):
    """Read a .spk.txt.bz2 or .tri.txt.bz2 file: two columns, time and electrode."""
    t, e = [], []
    with bz2.open(path, 'rt', errors='replace') as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    t.append(float(parts[0]))
                    e.append(int(float(parts[1])))
                except ValueError:
                    continue
    return np.asarray(t), np.asarray(e, dtype=int)


def recordings():
    """Stems of recordings that have both spikes and a stimulus log."""
    tri = sorted(glob.glob('**/stim/**/*.tri.txt.bz2', recursive=True))
    if not tri:
        tri = sorted(glob.glob('**/*.tri.txt.bz2', recursive=True))
    stems = [t[:-len('.tri.txt.bz2')] for t in tri]
    stems = [s for s in stems if os.path.exists(s + '.spk.txt.bz2')]
    random.Random(SEED).shuffle(stems)
    return stems


def prepare(stem):
    """Load one recording and drop stimulus sites with too few trials."""
    st, se = load(stem + '.spk.txt.bz2')
    tt, te = load(stem + '.tri.txt.bz2')
    order = np.argsort(st); st, se = st[order], se[order]
    order = np.argsort(tt); tt, te = tt[order], te[order]
    if len(st) < 500 or len(tt) < 200:
        raise ValueError('too little data')
    n_elec = int(se.max()) + 1
    labels, counts = np.unique(te, return_counts=True)
    keep_labels = labels[counts >= MIN_TRIALS]
    if len(keep_labels) < 5:
        raise ValueError('too few stimulus sites')
    mask = np.isin(te, keep_labels)
    return st, se, tt[mask], te[mask], n_elec, keep_labels


# ---------------------------------------------------------------- features

def response_grid(st, se, tt, te, n_elec, blank, window, drop=None):
    """Trials x electrodes x time bins of spike counts after each stimulus.

    drop: {stimulus label -> electrode index} whose channel is zeroed, used to
    show the result does not depend on the stimulated site recording itself.
    """
    binw = window / NBINS
    G = np.zeros((len(tt), n_elec, NBINS))
    lo = np.searchsorted(st, tt + blank)
    hi = np.searchsorted(st, tt + blank + window)
    for k in range(len(tt)):
        rt = st[lo[k]:hi[k]] - (tt[k] + blank)
        re = se[lo[k]:hi[k]]
        ok = (re >= 0) & (re < n_elec)
        rt, re = rt[ok], re[ok]
        if len(rt):
            b = np.clip((rt / binw).astype(int), 0, NBINS - 1)
            np.add.at(G[k], (re, b), 1.0)
        if drop is not None:
            d = drop.get(int(te[k]))
            if d is not None and 0 <= d < n_elec:
                G[k, d, :] = 0.0
    return G


def stim_site_map(st, se, tt, te, n_elec, labels):
    """For each stimulus label, the electrode responding most strongly early.

    Used to zero the stimulated site without assuming the stimulus index and
    the recording channel share a numbering scheme.
    """
    early = response_grid(st, se, tt, te, n_elec, 0.002, 0.010).sum(axis=2)
    return {int(l): int(early[te == l].mean(axis=0).argmax()) for l in labels}


def feature_sets(G):
    """RATE, NETWORK and SPATIAL representations of the same trials."""
    n_trials = G.shape[0]
    spatial = G.reshape(n_trials, -1)
    rate = G.sum(axis=(1, 2)).reshape(-1, 1)
    network = np.empty((n_trials, 10))
    for k in range(n_trials):
        g = G[k]
        per_elec = g.sum(axis=1)          # spikes per electrode
        per_bin = g.sum(axis=0)           # population time course
        n = per_elec.sum()
        active = float((per_elec > 0).sum())
        network[k] = [
            n,                                        # total spikes
            active,                                   # active electrodes
            active / g.shape[0],                      # active fraction
            (per_elec.max() / n) if n else 0.0,       # spatial concentration
            (per_bin.max() / n) if n else 0.0,        # temporal concentration
            float(np.argmax(per_bin)) if n else 0.0,  # busiest bin
            float(np.std(per_bin)),                   # time-course spread
            float(np.std(per_elec)),                  # across-electrode spread
            float((per_bin > 0).sum()),               # occupied bins
            float(per_elec.mean()),                   # mean per electrode
        ]
    return rate, network, spatial


# ---------------------------------------------------------------- scoring

def classifier(n_trees=300):
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=n_trees, random_state=SEED, n_jobs=-1))


def cv_score(X, y, n_trees=300):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_score(classifier(n_trees), X, y, cv=cv, n_jobs=1).mean()


def blocked_score(X, y, frac=0.7):
    """Train on the first 70% of the recording, test on the last 30%.

    Trials are in time order, so this also measures what a decoder frozen on an
    earlier period retains, which is the design a within-well study would depend on.
    """
    cut = int(len(y) * frac)
    model = classifier().fit(X[:cut], y[:cut])
    seen = np.isin(y[cut:], np.unique(y[:cut]))
    if seen.sum() == 0:
        return float('nan')
    return (model.predict(X[cut:][seen]) == y[cut:][seen]).mean()


def load_features(stem, blank=BLANK, window=WINDOW, drop_site=True):
    st, se, tt, te, n_elec, labels = prepare(stem)
    drop = stim_site_map(st, se, tt, te, n_elec, labels) if drop_site else None
    G = response_grid(st, se, tt, te, n_elec, blank, window, drop)
    rate, network, spatial = feature_sets(G)
    chance = 1.0 / len(labels)
    return rate, network, spatial, te, chance, (st, se, tt, n_elec)


# --------------------------------------------------------------- analyses

def run_compare(n_rec):
    """The headline comparison across independent recordings."""
    print('RATE vs NETWORK vs SPATIAL, window %d-%d ms, stimulated site excluded\n'
          % (BLANK * 1000, (BLANK + WINDOW) * 1000))
    print('%-38s %7s %7s %8s %8s' % ('RECORDING', 'CHANCE', 'RATE', 'NETWORK', 'SPATIAL'))
    print('-' * 72)
    rows = []
    for stem in recordings()[:n_rec]:
        try:
            rate, network, spatial, y, chance, _ = load_features(stem)
        except Exception:
            continue
        r, nw, sp = cv_score(rate, y), cv_score(network, y), cv_score(spatial, y)
        print('%-38s %6.1f%% %6.1f%% %7.1f%% %7.1f%%'
              % (os.path.relpath(stem)[-38:], 100 * chance, 100 * r, 100 * nw, 100 * sp))
        rows.append((chance, r, nw, sp))
    if not rows:
        print('No usable recordings found. Are you in the download directory?')
        return
    A = np.array(rows) * 100
    print('-' * 72)
    print('%-38s %6.1f%% %6.1f%% %7.1f%% %7.1f%%'
          % ('MEAN of %d' % len(rows), *A.mean(axis=0)))
    print('%-38s %6.1f  %6.1f  %7.1f  %7.1f ' % ('SD', *A.std(axis=0)))
    wins = int((A[:, 3] > A[:, 2]).sum())
    print('\nSPATIAL beat NETWORK in %d of %d recordings, mean margin %.1f points'
          % (wins, len(rows), (A[:, 3] - A[:, 2]).mean()))


def run_controls(stem=None):
    """Every way the headline number could be an artifact."""
    if stem is None:
        found = recordings()
        if not found:
            print('No usable recordings found. Are you in the download directory?')
            return
        stem = found[0]
    print('CONTROLS on %s\n' % os.path.relpath(stem))
    st, se, tt, te, n_elec, labels = prepare(stem)
    drop = stim_site_map(st, se, tt, te, n_elec, labels)
    chance = 100.0 / len(labels)
    print('trials %d, stimulus sites %d, chance %.1f%%\n' % (len(te), len(labels), chance))

    print('%-36s %9s %9s' % ('CONDITION', 'SHUF CV', 'BLOCKED'))
    print('-' * 56)

    def row(name, G, y):
        X = G.reshape(len(y), -1)
        print('%-36s %8.1f%% %8.1f%%' % (name, 100 * cv_score(X, y), 100 * blocked_score(X, y)))

    # 1. artifact: the blanking window gradient
    for blank in (0.002, 0.005, 0.010, 0.025, 0.050):
        G = response_grid(st, se, tt, te, n_elec, blank, 0.100)
        row('blank %4.0f ms, window 100 ms' % (blank * 1000), G, te)
    print()

    # 2. direct activation and the stimulated site
    row('late only, 20-100 ms',
        response_grid(st, se, tt, te, n_elec, 0.020, 0.080), te)
    row('stimulated channel zeroed',
        response_grid(st, se, tt, te, n_elec, 0.002, 0.100, drop), te)
    row('late AND channel zeroed',
        response_grid(st, se, tt, te, n_elec, 0.020, 0.080, drop), te)
    print()

    # 3. pipeline leakage
    rng = np.random.default_rng(SEED)
    row('permuted labels (want %.1f%%)' % chance,
        response_grid(st, se, tt, te, n_elec, 0.002, 0.100), rng.permutation(te))

    print('\nThe blanking gradient should fall then PLATEAU. A plateau beyond 25 ms')
    print('is longer than any artifact survives, so the floor is neural.')


def run_rate(n_rec):
    """Does firing rate predict decoding? Partial correlation controls for age."""
    print('FIRING RATE vs DECODING\n')
    print('%-32s %5s %8s %9s' % ('RECORDING', 'DIV', 'RATE', 'DECODE'))
    print('-' * 58)
    rows = []
    for stem in recordings()[:n_rec]:
        try:
            _, _, spatial, y, _, (st, se, tt, n_elec) = load_features(stem)
            div = int(os.path.basename(stem).split('-')[2])
        except Exception:
            continue
        duration = st.max() - st.min()
        rate = len(st) / duration / max(len(np.unique(se)), 1)
        acc = 100 * cv_score(spatial, y, n_trees=150)
        print('%-32s %5d %8.2f %8.1f%%' % (os.path.relpath(stem)[-32:], div, rate, acc))
        rows.append((stem, div, rate, acc))

    if len(rows) < 3:
        print('\nNot enough usable recordings. Are you in the download directory?')
        return
    names = [os.path.basename(r[0]) for r in rows]
    A = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
    div, rate, dec = A[:, 0], A[:, 1], A[:, 2]

    def partial(x, y, z):
        rx = x - np.polyval(np.polyfit(z, x, 1), z)
        ry = y - np.polyval(np.polyfit(z, y, 1), z)
        return np.corrcoef(rx, ry)[0, 1]

    print('-' * 58)
    print('\nn = %d' % len(rows))
    print('corr(rate, decode)            = %+.2f' % np.corrcoef(rate, dec)[0, 1])
    print('corr(DIV, rate)               = %+.2f' % np.corrcoef(div, rate)[0, 1])
    print('corr(DIV, decode)             = %+.2f' % np.corrcoef(div, dec)[0, 1])
    p = partial(rate, dec, div)
    print('partial corr(rate, decode|DIV) = %+.2f   -> r2 = %.2f' % (p, p ** 2))

    print('\nRATE-MATCHED PAIRS (rate within 10%, decoding gap > 20 points)')
    shown = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            ri, rj = rate[i], rate[j]
            if min(ri, rj) <= 0:
                continue
            if abs(ri - rj) / max(ri, rj) < 0.10 and abs(dec[i] - dec[j]) > 20:
                print('  %-20s %-20s rate %5.2f  gap %5.1f'
                      % (names[i], names[j], (ri + rj) / 2, abs(dec[i] - dec[j])))
                shown += 1
                if shown >= 15:
                    break
        if shown >= 15:
            break
    print('\n%d pairs. Equal firing rate, unequal function.' % shown)


def run_power(n_rec):
    """Paired endpoint variability, untreated drift, and wells per arm."""
    print('FROZEN DECODER ON UNTREATED DATA: the null distribution of the endpoint\n')
    print('%-32s %9s %9s %9s' % ('RECORDING', 'PRE', 'POST', 'DELTA'))
    print('-' * 62)
    pre_l, delta_l = [], []
    for stem in recordings()[:n_rec]:
        try:
            _, _, X, y, _, _ = load_features(stem)
        except Exception:
            continue
        n = len(y)
        a, b = int(n * 0.40), int(n * 0.50)
        seen = np.unique(y[:a])
        m_ho = np.isin(y[a:b], seen)
        m_po = np.isin(y[b:], seen)
        if m_ho.sum() < 20 or m_po.sum() < 20:
            continue
        model = classifier(200).fit(X[:a], y[:a])
        pre = 100 * (model.predict(X[a:b][m_ho]) == y[a:b][m_ho]).mean()
        post = 100 * (model.predict(X[b:][m_po]) == y[b:][m_po]).mean()
        print('%-32s %8.1f%% %8.1f%% %+8.1f'
              % (os.path.relpath(stem)[-32:], pre, post, post - pre))
        pre_l.append(pre)
        delta_l.append(post - pre)

    if len(delta_l) < 3:
        print('\nNot enough usable recordings.')
        return
    sd_abs = float(np.std(pre_l, ddof=1))
    sd_delta = float(np.std(delta_l, ddof=1))
    mean_delta = float(np.mean(delta_l))
    print('-' * 62)
    print('\nSD of absolute accuracy        = %.1f points' % sd_abs)
    print('SD of within-recording change  = %.1f points' % sd_delta)
    print('mean change                    = %+.1f points   (no compound was given)'
          % mean_delta)
    if abs(mean_delta) > 1:
        print('\nAccuracy drifts downward with no drug present. Any endpoint built on')
        print('this measure has to be vehicle-corrected or it will report suppression')
        print('that was only time.')

    z = (1.96 + 0.8416) ** 2
    print('\n%-16s %16s %14s' % ('DETECT', 'BETWEEN-ARM n', 'PAIRED n'))
    for d in (5, 10, 15, 20, 25):
        print('%-16s %13d  %13d'
              % ('%d points' % d,
                 math.ceil(2 * z * sd_abs ** 2 / d ** 2),
                 math.ceil(z * sd_delta ** 2 / d ** 2)))
    print('\nAt 6 wells per arm: between-arm detects ~%.0f points, paired ~%.0f points.'
          % ((1.96 + 0.8416) * sd_abs * math.sqrt(2.0 / 6),
             (1.96 + 0.8416) * sd_delta / math.sqrt(6)))
    print('80%% power, alpha 0.05.')


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('analysis',
                    choices=['compare', 'controls', 'rate', 'power', 'all'])
    ap.add_argument('-n', type=int, default=24,
                    help='recordings to use (default 24)')
    ap.add_argument('--stem', default=None,
                    help='specific recording for the controls analysis')
    args = ap.parse_args()

    jobs = ['compare', 'controls', 'rate', 'power'] if args.analysis == 'all' \
        else [args.analysis]
    for job in jobs:
        print('\n' + '=' * 72)
        print(job.upper())
        print('=' * 72)
        if job == 'compare':
            run_compare(args.n)
        elif job == 'controls':
            run_controls(args.stem)
        elif job == 'rate':
            run_rate(max(args.n, 40))
        elif job == 'power':
            run_power(min(args.n, 14))


if __name__ == '__main__':
    main()
