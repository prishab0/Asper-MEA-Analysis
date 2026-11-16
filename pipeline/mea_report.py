#!/usr/bin/env python3
"""Plate-aware well-level report for MEA drug experiments.

Input: one row per well, as a CSV, with a plate column, a group column
(treated vs control), a well name such as A2 (row letter + column number),
and numeric feature columns.

    python mea_report.py wells.csv --plate-col div --group-col drug --treated bic \
        --rate-col mfr --shape-cols s1,s2 --other-cols nb,dur,ibi

What it does, in order:
  1. QC: drops wells with too few bursts, if a burst-count column is given.
  2. Layout check: flags plates where plate position predicts treatment.
  3. Direction check: per-plate AUC for every feature, and whether it agrees across plates.
  4. Cross-plate transfer: train on one plate, test on another (features z-scored within plate).
  5. Leave-one-well-out accuracy with a permutation null (labels permuted within plate).
  6. A plain verdict with warnings.

It does not remove a layout confound. It tells you when one is present.
"""
import argparse, re, sys, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict


def parse_well(w):
    m = re.match(r'^([A-Za-z]+)(\d+)$', str(w).strip())
    if not m:
        return np.nan, np.nan
    return ord(m.group(1).upper()[0]) - 64, int(m.group(2))


def zscore_within(df, plate_col, cols):
    out = df.copy()
    for _, idx in df.groupby(plate_col).groups.items():
        sub = df.loc[idx, cols]
        sd = sub.std(ddof=0).replace(0, 1.0)
        out.loc[idx, cols] = (sub - sub.mean()) / sd
    return out


def safe_auc(y, x):
    return roc_auc_score(y, x) if len(set(y)) == 2 else np.nan


def layout_check(df, plate_col, y, well_col):
    notes = []
    rc = df[well_col].map(parse_well)
    df = df.assign(_row=[a for a, b in rc], _col=[b for a, b in rc], _y=y)
    for p, g in df.groupby(plate_col):
        if g['_row'].isna().all():
            notes.append(f'plate {p}: well names not parseable, layout not checked')
            continue
        a_row = safe_auc(g['_y'], g['_row'])
        a_col = safe_auc(g['_y'], g['_col'])
        t_rows = sorted(set(g.loc[g['_y'] == 1, '_row'].astype(int)))
        c_rows = sorted(set(g.loc[g['_y'] == 0, '_row'].astype(int)))
        disjoint = not (set(t_rows) & set(c_rows))
        flag = disjoint or max(abs(a_row - .5), abs(a_col - .5)) >= .4
        notes.append((flag, f'plate {p}: treated rows {t_rows}, control rows {c_rows}; '
                            f'AUC of row {a_row:.2f}, of column {a_col:.2f}'
                            + ('  <-- POSITION PREDICTS TREATMENT' if flag else '')))
    return notes


def lowo(X, y, plate_ids, perms=500, seed=0):
    rng = np.random.default_rng(seed)
    model = LogisticRegression(max_iter=1000)
    pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
    obs = balanced_accuracy_score(y, pred)
    null = []
    for _ in range(perms):
        yp = y.copy()
        for p in np.unique(plate_ids):
            m = plate_ids == p
            yp[m] = rng.permutation(y[m])
        null.append(balanced_accuracy_score(yp, cross_val_predict(model, X, yp, cv=LeaveOneOut())))
    null = np.array(null)
    return obs, (np.sum(null >= obs) + 1) / (perms + 1)


def transfer(Z, y, plate_ids, cols):
    res = []
    for a, b in itertools.permutations(np.unique(plate_ids), 2):
        tr, te = plate_ids == a, plate_ids == b
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        m = LogisticRegression(max_iter=1000).fit(Z.loc[tr, cols].values, y[tr])
        res.append((a, b, roc_auc_score(y[te], m.predict_proba(Z.loc[te, cols].values)[:, 1])))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--plate-col', required=True)
    ap.add_argument('--group-col', required=True)
    ap.add_argument('--treated', required=True, help='value in group column that marks treated wells')
    ap.add_argument('--well-col', default='well')
    ap.add_argument('--rate-col', required=True)
    ap.add_argument('--shape-cols', required=True)
    ap.add_argument('--other-cols', default='')
    ap.add_argument('--bursts-col', default=None)
    ap.add_argument('--min-bursts', type=int, default=10)
    ap.add_argument('--perms', type=int, default=500)
    ap.add_argument('--out', default=None, help='write the report to this file as well')
    a = ap.parse_args(argv)

    df = pd.read_csv(a.csv)
    lines = []
    P = lambda s='': lines.append(s)
    shape = [c for c in a.shape_cols.split(',') if c]
    other = [c for c in a.other_cols.split(',') if c]
    feats = [a.rate_col] + shape + other
    df[feats] = df[feats].astype(float)
    n0 = len(df)
    if a.bursts_col:
        df = df[df[a.bursts_col] >= a.min_bursts].reset_index(drop=True)
    P(f'MEA WELL-LEVEL REPORT  ({a.csv})')
    P(f'wells: {n0} read, {len(df)} kept after QC')
    y = (df[a.group_col].astype(str) == str(a.treated)).astype(int).values
    plates = df[a.plate_col].values
    P(f'treated: {int(y.sum())}, control: {int((1 - y).sum())}, plates: {sorted(set(plates.tolist()))}')
    warn = []

    P('\n1. LAYOUT CHECK')
    for n in layout_check(df, a.plate_col, y, a.well_col):
        if isinstance(n, tuple):
            P('  ' + n[1])
            if n[0]:
                warn.append('plate position predicts treatment, so treatment effects cannot be separated from position')
        else:
            P('  ' + n)

    P('\n2. DIRECTION CHECK  (AUC treated vs control per plate; 0.5 = no difference)')
    P(f'  {"feature":10s}' + ''.join(f'{str(p):>9s}' for p in sorted(set(plates.tolist()))) + '   agree?')
    flips = []
    for f in feats:
        aucs = [safe_auc(y[plates == p], df.loc[plates == p, f]) for p in sorted(set(plates.tolist()))]
        ok = all((x - .5) * (aucs[0] - .5) > 0 for x in aucs if not np.isnan(x))
        if not ok:
            flips.append(f)
        P(f'  {f:10s}' + ''.join(f'{x:9.3f}' for x in aucs) + ('   yes' if ok else '   NO (flips)'))
    if flips:
        warn.append('direction flips between plates for: ' + ', '.join(flips))

    Z = zscore_within(df, a.plate_col, feats)
    P('\n3. CROSS-PLATE TRANSFER  (train on one plate, test on another; AUC)')
    for name, cols in [('rate only', [a.rate_col]), ('shape only', shape)]:
        for tr, te, auc in transfer(Z, y, plates, cols):
            P(f'  {name:11s} {tr} -> {te}: {auc:.3f}')

    P('\n4. LEAVE-ONE-WELL-OUT  (balanced accuracy, labels permuted within plate)')
    for name, cols in [('rate only', [a.rate_col]), ('shape only', shape)]:
        acc, p = lowo(Z[cols].values, y.copy(), plates, perms=a.perms)
        P(f'  {name:11s} {acc:.3f}  permutation p = {p:.4f}')

    P('\n5. VERDICT')
    if warn:
        for w in sorted(set(warn)):
            P('  WARNING: ' + w)
    else:
        P('  No layout or direction warnings.')
    P('  This report cannot show that a readout separates rescue from suppression; that needs')
    P('  drug arms matched on firing rate with randomized plate layout and vehicle correction.')
    txt = '\n'.join(lines)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
