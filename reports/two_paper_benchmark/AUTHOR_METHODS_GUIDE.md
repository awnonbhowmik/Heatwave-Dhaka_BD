# Author methods guide

The target asks whether all three future Tmax observations exceed the chosen threshold. Issue time is the end of today; a lead of one day means tomorrow starts the three-day target window. This differs from retrospectively marking every day in an already known event and from predicting a single hot day.

Leakage occurs whenever a feature, transform, label, threshold, calibration map, or decision cutoff uses information unavailable at its fitting/issue time. Here lags are built on the continuous calendar, all learned preprocessing is refit inside training folds, and labels must be known before fitting. Observed temperatures through today are valid predictors; future temperatures are not.

Daily windows overlap, and many positives represent the same physical spell. Chronological season blocks protect the forecast ordering, while held-out-year resampling is more defensible than treating rows as independent. Even so, only eleven evaluation seasons and few distinct events sharply constrain complexity and certainty.

Class imbalance makes accuracy misleading: predicting no event can be highly accurate. Recall measures captured positive windows; precision measures how many alerts were correct; specificity measures rejected negatives; average precision summarizes ranking under rarity. Baselines show whether ML adds value beyond seasonality and observed temperature transitions.

Calibration asks whether predicted probabilities correspond to observed frequencies. Brier score combines calibration and discrimination and is not a calibration-only statistic. Sigmoid maps and alert thresholds are learned from chronologically held-out training predictions. A 0.5 operating point is retained only as sensitivity.

SHAP decomposes an uncalibrated tree output for a particular fitted model. It does not identify causes, correlated predictors can divide or exchange attribution, and a weak model's attributions do not establish physical heatwave mechanisms. Block permutation is also diagnostic and deliberately disrupts feature dependence.

The earlier GEE analysis estimates adjusted historical associations. This benchmark evaluates future-aligned predictions. Association can exist without useful forecast skill, and predictive importance can exist without causal interpretation.
