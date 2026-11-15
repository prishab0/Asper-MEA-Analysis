# Prediction sheet and number audit (draft, not pre-registered until posted)

Author: Prisha Bhanushali. Status: exploratory, simulation is a hand-built model, not validation.

## 1. What the model now does (net3.py)
Spiking network, 240 E + 60 I, short-term depression plus a slow NMDA-like current.
Calibrated by hand to your real plates: control gives about 13 Hz mean rate, bursts about 0.7 s wide, 6 to 8 s apart, burst fraction about 0.06. Real wells: bursts about 1 s, interval 4 to 12 s.

Parameter sensitivity is high. Only a narrow window of drive and coupling gives bursting at all; most settings are silent or saturate. Treat every number below as a model property, not a biological estimate.

## 2. Model results (4 seeds, 90 s each)
| Arm | Rate (Hz) | Bursts | Width (s) | Interval (s) |
|---|---|---|---|---|
| control | 12.6 | 6.2 | 0.74 | 8.4 |
| GABA block 50% | 28.1 | 13.0 | 0.56 | 7.1 |
| GABA block 90% | 78.1 | 1 (tonic) | n/a | n/a |
| drive -1 mV | 2.6 | 1.2 | 0.12 | 8.3 |
| drive +1 mV | 14.1 | 1 | n/a | n/a |

Reading: partial GABA block raises rate and burst count and shortens bursts; strong block tips the network into tonic firing with no bursts. Small drive changes (+/-1 mV) destroy bursting. That fragility is itself a finding: aggregate burst features in this model are unstable, which fits the real plates where they flipped direction.

## 3. Prediction sheet (to test in the lab or on new public data)
Hypothesis H1: a compound that matches the aggregate firing rate of a reference can still be separated by burst-shape or evoked-response features.
Arms: (a) reference rescue compound, (b) rate-matched suppression by reduced drive, (c) rate-matched suppression by reduced excitation, (d) rate-matched suppression by raised inhibition, (e) vehicle.
Readouts: mean rate, burst count, width, interval, shape coordinates s1 and s2, evoked decoder accuracy.
Design: at least 24 wells per arm, 2 plates from different preparations, drug and vehicle wells randomized across rows (the existing data confound all bicuculline wells into rows A and B).
Falsified if: (1) shape features do not separate arms at AUC above 0.7 in both plates after rate matching; (2) direction of any shape feature flips between plates; (3) separation disappears when train and test come from different cultures.
Not claimed: that these readouts distinguish rescue from suppression. That is untested.
Wells needed: for the decoder-accuracy endpoint, public untreated recordings give a between-recording SD of 24.1 points and a within-recording SD of 5.9 points (evoked_decoder.py power). At 80% power, a between-arm design needs about 92 wells per arm for a 10-point change; a paired pre/post design needs about 3. Accuracy also drifted down by 4.3 points with no drug present, so every endpoint must be vehicle-corrected against time-matched controls.

## 4. Claim to source audit
| Claim in manuscript | Value | Source | Status |
|---|---|---|---|
| Recording-level decoder accuracy | 36.8% | evoked_decoder.py compare, mean of 24 (rate 4.8, network 7.6, chance 1.8) | RESOLVED: manuscript and one-pagers corrected from 36.5 to 36.8 |
| Culture-level accuracy | 39.9% [28.7, 52.0] | culture_level.py (14 cultures; rate 5.4, network 8.5) | Reproduced from the rounded table: 39.9 [28.8, 51.9]; interval differs by 0.1 from rounding |
| Rate-only / network decoder | 5.4% / 8.5% | decoder output | matches summary |
| Chance | 1.8% | 1/n classes | check class count |
| Spatial beats network | 14 of 14 cultures, min margin 5.9 points | culture_level.py | RESOLVED: manuscript said "rate"; the 5.9 margin is against network, text corrected |
| Transfer | 36.2 / 7.7 / 5.7% | transfer test | matches summary |
| Plate replication | c1 AUC 1.000, p 2.4e-05, n=24 per plate (14 control, 10 bic) | frozen_wells.csv | matches; note the row confound |
| Shape AUCs | 1.000 / 0.878 / 0.771 | revised feature set | confirm they are the plate numbers in Table 1 |
| Aggregate features flip across plates | rate, count, duration, interval, fraction flip; s2 consistent | real_wells.csv, 48 wells | NEW: add to manuscript if you revise |
| s1 | consistent direction but AUC 0.057 vs 0.321 | real_wells.csv | weak, state as direction only |

Open items: line up each manuscript number against a script output file with the commit hash; the 36.5 vs 36.8 discrepancy needs one canonical figure; "Competing interests: none" and "no external funding" must stay true (EF $10,000 grant, any company use).

Open: the 36.5% vs 35.8% frozen-split control in the manuscript is a separate single-recording check and was left unchanged. The 'rate-matched pairs' output of evoked_decoder.py compares different cultures and ages and should not be cited as evidence about function.
