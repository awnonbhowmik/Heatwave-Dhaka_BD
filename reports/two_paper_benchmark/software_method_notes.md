# Software method notes

The implementation follows the installed APIs and official documentation for:

- [scikit-learn regularized logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), using the binary-compatible `liblinear` solver for L1/L2 candidates;
- [imbalanced-learn balanced random forest](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html), whose per-tree sampling balances the classes;
- [XGBoost's scikit-learn estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html), with a weighted binary logistic objective and capped CPU threads;
- [SHAP TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html), using a training-only background and the interventional feature-dependence option.

The recorded environment, estimator parameters, warnings, seeds, thread caps, and additivity checks are saved under `results/two_paper_benchmark/metadata`, `tuning`, and `explanations`. SHAP output scales are recorded per fold because scikit-learn forest binary outputs and XGBoost raw margins are not interchangeable.

Runtime compatibility note: the initially pinned SHAP 0.49.1 could not parse XGBoost 3.1.1's vector-valued serialized `base_score` (`ValueError: could not convert string to float: '[5E-1]'`). No estimator was substituted. SHAP was upgraded to 0.52.0, the failure was retained in the execution history, and the explanation stage was rerun with additivity validation.
