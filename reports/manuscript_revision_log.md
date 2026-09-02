# Manuscript revision log

| Revision | Trigger | Material change | Files affected | Claim impact |
|---|---|---|---|---|
| 1 | Reproducibility audit | Replaced workflow-first framing with an original-article structure centered on warming, definition dependence, NB2 count inference, and antecedent associations | `manuscript/original_article_clean.md`; article tables and figures | Establishes the article's primary question and prespecified outcome |
| 2 | Meteorological review | Removed the unsupported station point, exposed missing provenance and homogenization metadata, and clarified partial-calendar-year versus complete hot-season coverage | Methods, limitations, Figure 1 and caption | Prevents unsupported spatial and homogeneity claims |
| 3 | Statistical review | Replaced Poisson-based diagnostics with selected-NB2 residual and influence diagnostics | Count-model code, Table 5, Figure 6, supplement | Makes adequacy checks consistent with model selection |
| 4 | Statistical review | Expanded complete-year/hot-season sensitivity and formally compared \(T_{\min}\) and \(T_{\max}\) slopes | Table 4, Figure 4, Results | Removes the unsupported claim that minimum temperature warmed faster |
| 5 | Association review | Excluded target-derived variables, retained four antecedent weather predictors, used year-clustered GEE, and added onset, lag-window, and influential-season sensitivity | Table 7, Figure 7, Results and Discussion | Restricts inference to temporally ordered adjusted associations |
| 6 | Validation review | Changed rolling validation from GLM to the same GEE estimator used in the article, with training-only scaling | Association code, Table 8, Figure 7, supplement | Provides estimator-aligned out-of-sample evidence |
| 7 | Calibration review | Added PR-AUC, Brier score, calibration intercept/slope, sensitivity, specificity, and zero-positive-season handling | Table 8, Figure 7, Results | Shows improved ranking but systematic underprediction; removes operational-warning claim |
| 8 | Multiplicity review | Applied BH only to the four prespecified weather coefficients and separately across six definition trends | Tables 3 and 7, abstract, Results, Discussion, Conclusions | Reclassifies compound result as exploratory: nominal \(p=0.0209\), BH \(q=0.126\) |
| 9 | Forecasting review | Demoted forecasting to the supplement and removed exact future heatwave-count claims | Supplement and Discussion | Limits claims to validated monthly-temperature comparisons |
| 10 | Figure review | Kept Figure 3 title-free, reduced matrix spacing, and matched colorbar height to matrix height | Figure 3 plotting code and output | Improves publication layout without changing inference |
| 11 | Deliverable review | Added editable clean, yellow-highlighted, and supplementary Word builds with native tables and captions below figures | `scripts/build_manuscripts.py`; `.docx` outputs | Produces submission-ready editable artifacts |

No reviewer-requested change was rejected. The unresolved station metadata and author declarations remain explicit pre-submission actions rather than being inferred from the dataset.
