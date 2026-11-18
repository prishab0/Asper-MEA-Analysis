# Simulated plates: rescue vs rate-matched suppression (SIMULATED, not data)
run_arms.py builds 4 plates x 24 wells with net3.py (arms: disease = inhibition halved, rescue = inhibition restored, two suppression mechanisms tuned to the control firing rate). analyze_arms.py fits logistic regression and random forest (each plate held out) and mixed models (feature ~ arm + (1|plate)). Output: arms_analysis_output.txt.
Limits: rescue is exactly the control parameter set, the separation is a built-in consequence of the model, layout is randomised, and the model only bursts in a narrow parameter range. It shows what the analysis can and cannot detect, not biology.
Run: pip install brian2 statsmodels scikit-learn pandas scipy; python run_arms.py; python analyze_arms.py
