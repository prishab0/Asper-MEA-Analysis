# MEA well-level report

One command that checks a plate-based MEA drug experiment for the problems that fooled us: plate position that predicts treatment, features that flip direction between plates, and accuracy that disappears across plates.

    pip install -r requirements.txt
    python mea_report.py wells.csv --plate-col div --group-col drug --treated bic \
        --rate-col mfr --shape-cols s1,s2 --other-cols nb,dur,ibi,frac --bursts-col nb

Input: one row per well. Needs a plate column, a group column, a well name like A2, a firing-rate column, and one or more shape columns. See `example_report_day17_day18.txt` for output on the public bicuculline plates (`real_wells.csv` in `simulations/spiking_network/`).

Checks: layout (AUC of row and column against treatment), per-plate direction of every feature, cross-plate transfer, leave-one-well-out accuracy with a permutation null that shuffles labels within plate.

Limits: it reports a layout confound but cannot remove it. It does not do vehicle-drift correction yet. A passing report is not evidence that a readout separates rescue from suppression. Tests: `python -m pytest tests`.
