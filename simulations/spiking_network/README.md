# Spiking-network simulation: can burst shape separate rescue from rate-matched suppression?

Model assumptions only, no data. 400 leaky integrate-and-fire neurons (320 excitatory, 80 inhibitory), depressing
excitatory synapses, background noise. Bursts emerge from recurrent excitation plus depression.

Arms (24 simulated wells each, 30 s, parameter jitter per well):
- rescue: healthy parameters
- disease: raised excitatory coupling and reduced inhibition (about 1.8x the rescued firing rate)
- three suppression arms that keep the diseased network but are each calibrated to the rescued mean rate:
  lower background drive, weaker excitatory coupling, stronger inhibition

Run order: `calib.py` (calibrates), `run_wells.py 24` (simulates), `analyze.py` and `fig.py`.
Requires brian2, numpy, scipy, scikit-learn, matplotlib. Seeds are fixed.

This is a test of whether a readout CAN separate two states in a model. It does not show that real cultures behave this way.
