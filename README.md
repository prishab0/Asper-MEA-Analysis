[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24238927.svg)](https://doi.org/10.5281/zenodo.24238927)

# Asper Labs: public MEA reanalysis

Code behind *Asper Labs: public MEA reanalysis* (v7, November 2026; PDF in `docs/`).

The question: does a representation of the spatiotemporal response, or of burst shape, carry compound-specific information that aggregate measures such as mean firing rate discard? Everything here runs on **public** data. No new recordings were generated.

Contact: Prisha Bhanushali, prb330@lehigh.edu. Comments and corrections are welcome.

## Status

This is exploratory work, not a validated result. In particular:

- The central claim, that a readout can separate functional rescue from nonspecific suppression, is **untested**. No analysis here has drug conditions matched on firing rate.
- Plate position is confounded with treatment in the bicuculline data. Well-level validation does not remove that confound.
- A preliminary human iPSC result was withdrawn pending a well-level re-analysis and is not part of this repository.
- The evoked-response decoder is in `scripts/evoked_decoder.py`. The frozen-embedding replication (`scripts/run_frozen.py`) and the placebo (`scripts/placebo.py`) are reported in the v7 PDF. Both plates share one layout, so replication across plates does not remove the position confound.

## Layout

| Path | What it does |
|---|---|
| `scripts/run_day17.py` | Second bicuculline plate (day 17): joint embedding with day 18, cross-plate test. Writes `day17_day18_wells.csv`. |
| `scripts/run_robustness.py` | Sweeps minimum burst size, number of shape coordinates and rate-range width, with permutation tests and bootstrap intervals. |
| `scripts/run_frozen.py` | Frozen-embedding check (embedding fitted on day 18 only) and a plate-position check. |
| `scripts/wellcv.py` | Leave-one-well-out validation with a well-level permutation null, bootstrap intervals over wells and a firing-rate-only comparison. |
| `scripts/placebo.py` | Placebo test on control wells only, with labels based on plate position. |
| `scripts/null_check.py` | Simulation showing that a leave-one-out permutation null falls below 1/k when classes are small. |
| `scripts/run_lamberti.py`, `run_lamberti_windows.py` | Paired baseline versus APV comparison, and a rate-matched window check. |
| `simulations/rescue_power.py` | Power simulation for a chronic-dose rescue design. **Every effect size and variance in it is an assumption.** |
| `figures/` | Scripts and images for the figures in the report. Values are hardcoded from the analysis outputs. |
| `tests/make_fake_lamberti.py` | Builds a synthetic folder in the layout of the Lamberti deposit, to test the scripts without the data. |

## Setup

```
pip install -r requirements.txt
git clone https://github.com/LevinaLab/burst-shape
```

The `run_*.py` scripts import the `burst_shape` package and are run from inside the `burst-shape` folder, for example `uv run python -u run_day17.py`. Copy the scripts there first. `run_day17.py` downloads the day 17 and day 18 spike files from figshare and needs about 2 GB of free disk. `scripts/wellcv.py`, `placebo.py`, `null_check.py` and `simulations/rescue_power.py` need only numpy, pandas, SciPy and scikit-learn.

## Well-level validation

```
python scripts/wellcv.py day17_day18_wells.csv --features shape_1,shape_2 --rate-col MFR --perms 2000
python scripts/placebo.py
```

`--features` is required. Do **not** pass plate position (`well_idx`) or plate day (`div`) as features. In this layout all bicuculline wells sit in fixed positions, so those columns encode the label and give a perfect score that reflects position, not biology. An early run that included them scored 1.000 for that reason and was discarded.

Chance is 1/k for k equal classes. With few wells per class, a leave-one-out permutation null falls below 1/k (see `scripts/null_check.py`), so report the null's 95th percentile and a bootstrap interval over wells, not a single "chance" number. A perfect score gives a bootstrap interval of [1, 1]; that reflects a small sample, not certainty.

## Data

All public; see `REFERENCES.md`. Raw data are not stored here.

## Licence

MIT (`LICENSE`). The burst-shape package is separately licensed by its authors.
