# Frozen prediction contract

Protocol frozen: 2026-09-06 UTC, before inspecting new benchmark model rankings. This is a frozen reanalysis protocol, not a prospective registration, because this dataset has already been examined in earlier project analyses.

## Question and issue time

The primary question is: “Does meteorological history improve prediction of a forthcoming persistent hot spell beyond seasonality and recent temperature alone, and how does performance change with lead time?”

Issue time is the end of day *t*, after that day's measurements are assumed available. That is explicitly a retrospective availability assumption until source/product release latency is documented. Predictors use the continuous-calendar window *t−6* through *t*; February observations may therefore inform early-March issues.

## Outcomes

The primary endpoint is `y(t,h)=1` when Tmax is at least 36 °C on all three dates *t+h*, *t+h+1*, and *t+h+2*. It is zero only when all three target measurements are known and the condition is not met; an absent or missing target date produces an unknown label. The three-day target window must fall wholly within March–June.

The primary lead is `h=1` day. Secondary direct, separately fitted leads are 3 and 7 days. This is “prediction of a future three-day persistent-hot window,” not recursive weather forecasting, not a forecast of an entire retrospectively labelled heatwave, and not necessarily a new-event onset. Continuations of an existing spell are eligible.

For the h=1 onset-risk evaluation, issue dates are restricted to `Tmax(t)<36 °C`; a positive denotes a qualifying spell beginning tomorrow. Eligible days, distinct associated events, recall, precision, and false alerts are reported separately.

The required relative-threshold sensitivity applies a training-cutoff-specific 90th-percentile calendar-day threshold to all three future dates. It is evaluated separately from the fixed-threshold outcome. A 14-day lookback sensitivity is limited to the prespecified regularized logistic model.

## Information sets

`S0` contains target-date calendar terms and a prespecified linear time term. `S1` adds only issue-time-known Tmax/Tmin summaries and the observed current hot-day run length. `S2` adds parsimonious issue-time summaries of available humidity, precipitation, wind, pressure, cloud/radiation, and shallow-soil-moisture variables. Future temperatures, future event membership, and any feature timestamp after issue time are prohibited.

## Validation, calibration, and selection

Outer expanding-window training precedes each complete March–June test season from 2014 through 2024. Event-free seasons remain. Training samples whose target label was unavailable at the fitting cutoff are purged. Three expanding, year-blocked inner folds select among 12 reproducibly sampled candidates by scikit-learn average precision; no random row split is allowed.

Imputation, scaling, correlation screening, class weighting, hyperparameters, sigmoid calibration, and a balanced-accuracy threshold are estimated using training data or chronological cross-fitted training predictions only. Outer test labels are used only for evaluation. The 0.5 threshold is retained as a sensitivity. Undefined metrics remain missing with a reason.

Comparisons use held-out-year block resampling (2,000 draws), never an independent daily-row bootstrap. Explanations concern out-of-sample predictions from a tree model fitted before each explained season, use training-only backgrounds, and are predictive—not causal.

Every analytical row carries `issue_date`, `feature_start`, `feature_end`, `target_start`, `target_end`, `label_available_date`, `lead`, `outcome`, and associated fixed-threshold event ID. Split manifests and prediction rows preserve fold and model provenance.
