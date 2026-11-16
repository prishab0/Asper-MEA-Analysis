import numpy as np, pandas as pd, subprocess, sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def run(df, tmp_path, extra=()):
    p = tmp_path / 'w.csv'; df.to_csv(p, index=False)
    r = subprocess.run([sys.executable, os.path.join(HERE, 'mea_report.py'), str(p), '--plate-col', 'plate',
        '--group-col', 'grp', '--treated', 'T', '--rate-col', 'rate', '--shape-cols', 's1,s2', '--perms', '30', *extra],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout
def make(confounded, signal, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for plate in (1, 2):
        for i, row in enumerate('ABCD'):
            for c in range(1, 7):
                if confounded: t = row in 'AB'
                else: t = rng.random() < .5
                rows.append(dict(plate=plate, well=f'{row}{c}', grp='T' if t else 'C',
                    rate=rng.normal(5 * plate, 1), s1=rng.normal(signal * t, 1), s2=rng.normal(0, 1), nb=50))
    return pd.DataFrame(rows)
def test_confound_flagged(tmp_path):
    assert 'POSITION PREDICTS TREATMENT' in run(make(True, 0), tmp_path)
def test_randomised_layout_not_flagged(tmp_path):
    assert 'POSITION PREDICTS TREATMENT' not in run(make(False, 0, seed=3), tmp_path)
def test_signal_gives_high_shape_accuracy(tmp_path):
    out = run(make(False, 3.0, seed=1), tmp_path)
    line = [l for l in out.splitlines() if l.strip().startswith('shape only') and 'permutation' in l][0]
    assert float(line.split()[2]) > 0.8
def test_flip_detected(tmp_path):
    df = make(False, 0, seed=2)
    m = (df.plate == 2) & (df.grp == 'T'); df.loc[m, 'rate'] += 6
    m = (df.plate == 1) & (df.grp == 'T'); df.loc[m, 'rate'] -= 6
    assert 'direction flips' in run(df, tmp_path)
