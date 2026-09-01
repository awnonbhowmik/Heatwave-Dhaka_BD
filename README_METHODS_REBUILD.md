# Methods-first statistical rebuild

This branch replaces the article's descriptive/forecast-conversion core with a reproducible sequence: immutable-source audit, descriptive analysis, alternative event definitions, direct inference, diagnostics, chronological validation, and uncertainty. The legacy notebook and figures remain available but are not treated as validated results.

## Research questions

1. How have heatwave frequency, persistence, duration, and intensity changed under alternative definitions?
2. Which strictly antecedent weather conditions are associated with persistent heatwave days after seasonality, trend, collinearity, and temporal clustering are addressed?
3. How well do temperature and occurrence models perform on unseen hot seasons, and what uncertainty surrounds 2025–2029 conditional scenarios?

The primary event is **$T_{\max} \ge 36\,{}^\circ\mathrm{C}$ for at least three consecutive days**. The one-day operational threshold, two-day threshold, percentile definitions, compound day/night definition, alternate reference period, partial-2024 handling, and model classes are sensitivity analyses.

## Data and boundaries

The raw daily CSV, XLSX, Global Forest Watch file, and shapefiles are never modified. Complete-year annual analyses end in 2023. March–June 2024 is included only because all 122 expected dates are present. Tree-cover information is division-level ecological context. No urban–rural contrast, health outcome, or vulnerability measure is available; the analysis cannot make those causal or impact claims.

## Structure

- `config/analysis.yml`: prespecified seed, definitions, origins, and output settings.
- `src/heatwave_analysis/`: reusable data, event, modeling, diagnostic, forecast, plot, and report functions.
- `scripts/run_all.py`: one-command pipeline.
- `scripts/validate_outputs.py`: source, output, interval, and report consistency checks.
- `notebooks/`: thin ordered interfaces without hidden analytical logic.
- `tests/`: event, threshold, partial-year, split, leakage, interval, and consistency tests.
- `results/`: machine-readable tables, derived data, diagnostics, forecasts, and metadata.
- `reports/`: audit and manuscript-ready Markdown drafts.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
make analysis
make test
make validate
```

Direct equivalent:

```bash
MPLCONFIGDIR=/tmp/heatwave-mpl .venv/bin/python scripts/run_all.py --config config/analysis.yml
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_outputs.py --config config/analysis.yml
```

All figures are written as 300-DPI PNG plus vector PDF. All numbered tables are written as CSV plus publication-ready Markdown. `results/metadata/run_metadata.json` records software, seed, configuration, hashes, starting commit, time, and runtime.

## Interpretation boundaries

Correlations are exploratory. Adjusted odds ratios describe associations, not causes. Forecast models are ranked only through rolling-origin out-of-sample metrics on identical targets. The monthly forecast exercise cannot validate daily event sequences, so direct 2025–2029 count results are labeled trend-based conditional scenarios. If advanced models fail to beat naive baselines, the failure is retained and precise long-range forecasting is rejected.
