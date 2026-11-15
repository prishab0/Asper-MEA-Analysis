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
Wells needed: the real day 17 and day 18 data give control CV of 0.26 to 0.55 for rate, so detecting a 30% effect needs roughly 12 to 40 wells per arm; confirm with a power calculation on your own pilot.

## 4. Claim to source audit
| Claim in manuscript | Value | Source | Status |
|---|---|---|---|
| Recording-level decoder accuracy | 36.5% (36.8% in your rerun) | decoder script output | State both or the rerun value; confirm which the text uses |
| Culture-level accuracy | 39.9% [28.7, 52.0] | culture bootstrap, 10,000 resamples | matches summary |
| Rate-only / network decoder | 5.4% / 8.5% | decoder output | matches summary |
| Chance | 1.8% | 1/n classes | check class count |
| Spatial beats rate | 14 of 14 cultures, min margin 5.9 points | culture-level check | matches summary |
| Transfer | 36.2 / 7.7 / 5.7% | transfer test | matches summary |
| Plate replication | c1 AUC 1.000, p 2.4e-05, n=24 per plate (14 control, 10 bic) | frozen_wells.csv | matches; note the row confound |
| Shape AUCs | 1.000 / 0.878 / 0.771 | revised feature set | confirm they are the plate numbers in Table 1 |
| Aggregate features flip across plates | rate, count, duration, interval, fraction flip; s2 consistent | real_wells.csv, 48 wells | NEW: add to manuscript if you revise |
| s1 | consistent direction but AUC 0.057 vs 0.321 | real_wells.csv | weak, state as direction only |

Open items: line up each manuscript number against a script output file with the commit hash; the 36.5 vs 36.8 discrepancy needs one canonical figure; "Competing interests: none" and "no external funding" must stay true (EF $10,000 grant, any company use).
