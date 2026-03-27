# Heatwave Analysis — Dhaka, Bangladesh (1972–2024)

Analysis of 52 years of daily temperature records from Dhaka, Bangladesh, examining heatwave trends, novel heat-risk indicators, and multi-model forecasts to 2029. Accompanying code and figures for the manuscript *Heatwave Dynamics in Dhaka, Bangladesh: Trends, Drivers, and Projections*.

---

## Repository structure

```
Heatwave-Dhaka_BD/
├── data/
│   ├── 1972_2024_Heatwave_Daily.csv   # 19,316 daily records, 32 variables
│   └── GFW_Dhaka.csv                  # Global Forest Watch annual deforestation (2001–2023)
├── figures/
│   ├── fig1_overview.png              # Warming trend, heatwave days, deforestation
│   ├── fig2_temperature_structure.png # Daily range, seasonal cycle, Tmin vs Tmax
│   ├── fig3_heatwave_characteristics.png  # Annual counts, duration, monthly distribution
│   ├── fig4_heat_index.png            # NWS apparent temperature burden and trend
│   ├── fig5_recovery_gap.png          # Nighttime recovery gap (Tmax−Tmin) asymmetric warming
│   ├── fig6_compound_events.png       # Compound hot+dry events
│   ├── fig7_sarima_forecast.png       # SARIMA(3,1,0)×(1,0,0)₁₂ forecast 2025–2029
│   ├── fig8_arima_forecast.png        # ARIMA(5,1,0)×(1,0,0)₁₂ forecast 2025–2029
│   ├── fig9_lstm_forecast.png         # PyTorch LSTM forecast with 1-step test overlay
│   ├── fig10_correlation_matrix.png   # Climate drivers correlation matrix
│   ├── fig11_model_comparison.png     # 4-panel multi-model comparison summary
│   ├── supp_rf_forecast.png           # Supplementary: Random Forest (notebook only)
│   └── supp_xgboost_forecast.png      # Supplementary: XGBoost (notebook only)
├── analysis.ipynb                     # Full reproducible pipeline (see below)
├── HeatWave_V1.docx                   # Manuscript
└── README.md
```

---

## Dataset

**`data/1972_2024_Heatwave_Daily.csv`**
- 19,316 daily records: 1972-01-01 to 2024-11-18
- Key variables: `tmax` (daily max °C), `tmin` (daily min °C), `tmean`, `rh_mean` (relative humidity %), `sm_mean` (soil moisture), `vpd_mean` (vapour pressure deficit)
- Zero missing values in `tmax`

**`data/GFW_Dhaka.csv`**
- Global Forest Watch annual tree cover loss for the Dhaka division (BGD)
- 85,800 ha lost 2001–2023; Spearman ρ = 0.446 with annual mean Tmax (p = 0.033)

**Heatwave definition:** Tmax ≥ 36 °C — Bangladesh Meteorological Department standard.

---

## Key findings

### Temperature trends (1972–2024)
| Metric | Value |
|--------|-------|
| Mean Tmax | 30.14 °C |
| Record Tmax | 40.2 °C (9 May 2023) |
| Tmax warming rate | +0.204 °C / decade |
| Total Tmax warming | +1.06 °C |
| Tmin warming rate | +0.235 °C / decade *(faster than Tmax — UHI signal)* |

### Heatwave statistics
| Metric | Value |
|--------|-------|
| Total heatwave days | 377 |
| Historical mean | 7.1 days / year |
| Total events | 136 |
| Longest event | 15 days (19 April 2024) |
| Season concentration | 87% of days in April–May |

### Novel indicators
| Indicator | Finding |
|-----------|---------|
| Tmin warming faster than Tmax | +0.031 °C/decade difference (p < 0.05) — nighttime recovery worsening |
| Heat Index 95th-pct exceedance days | Significant upward trend (p < 0.001) |
| Compound hot+dry events | 300 total, 5.7/yr — clustering pre-monsoon, trend not yet detectable |
| Deforestation–temperature link | ρ = 0.446 (p = 0.033) — contextual co-variation |

### Forecast summary (SARIMA primary model, 2025–2029)
| Year | Projected Tmax | Projected HW days |
|------|---------------|-------------------|
| 2025 | 31.3 °C | ~20 |
| 2026 | 31.6 °C | ~24 |
| 2027 | 31.9 °C | ~27 |
| 2028 | 32.2 °C | ~30 |
| 2029 | 32.6 °C | ~35 |
| **5-yr mean** | **31.9 °C** | **~27 days/yr** |

Historical mean: 7.1 days/yr → projected **~280% increase** by 2029.

---

## Figures

| # | File | Description |
|---|------|-------------|
| 1 | `fig1_overview.png` | 3-panel: Tmax warming trend · annual heatwave days · deforestation overlay |
| 2 | `fig2_temperature_structure.png` | Daily temperature range, seasonal cycles, Tmin vs Tmax trend decomposition |
| 3 | `fig3_heatwave_characteristics.png` | Annual heatwave day counts · monthly distribution · event duration histogram |
| 4 | `fig4_heatwave_calendar.png` | Year–month heatmap (seaborn) showing April–May concentration across all 52 years |
| 5 | `fig5_heat_index.png` | NWS apparent temperature trend · HI burden on heatwave days · extreme HI exceedance |
| 6 | `fig6_recovery_gap.png` | Nighttime recovery gap (Tmax−Tmin) asymmetric warming — novel UHI indicator |
| 7 | `fig7_compound_events.png` | Compound hot+dry event frequency, calendar heatmap, seasonal distribution |
| 8 | `fig8_sarima_forecast.png` | **Primary forecast** — SARIMA(3,1,0)×(1,0,0)₁₂ · AIC=2109.7 · 95% CI shown |
| 9 | `fig9_arima_forecast.png` | ARIMA(5,1,0)×(1,0,0)₁₂ comparison · AIC=2159.1 · 95% CI shown |
| 10 | `fig10_lstm_forecast.png` | PyTorch LSTM (2-layer, 64 hidden) · test R²=0.846 · 1-step fit + iterative forecast |
| 11 | `fig11_correlation_matrix.png` | Pearson correlation matrix of all climate drivers |
| 12 | `fig12_model_comparison.png` | 4-panel summary: temperature trajectories · HW day projections · AIC bars · ML metrics |

---

## Reproducing the analysis

### Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

### Setup

```bash
conda create -n heatwave python=3.11
conda activate heatwave
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn xgboost torch jupyter ipykernel
python -m ipykernel install --user --name heatwave
```

### Run

```bash
cd Heatwave-Dhaka_BD
jupyter notebook analysis.ipynb
```

Select kernel **heatwave** and run all cells (Kernel → Restart & Run All). Execution takes ~5–8 minutes (LSTM training on CPU).

All figures are saved to `figures/` at 300 DPI automatically.

---

## Models used

| Model | Order | AIC | Best for |
|-------|-------|-----|----------|
| SARIMA | (3,1,0)×(1,0,0)₁₂ | 2109.7 | **Primary forecast** — captures monsoon seasonality |
| ARIMA | (5,1,0)×(1,0,0)₁₂ | 2159.1 | Comparison baseline |
| LSTM | 2-layer, 64 hidden, SEQ=24 | — | 1-step fit (R²=0.846); iterative forecast reverts to trend mean |
| Random Forest | 500 trees, depth=4 | — | Supplementary only — insufficient annual training data |
| XGBoost | 300 estimators, depth=3 | — | Supplementary only — same limitation as RF |

---

## Citation

> [Authors]. Heatwave Dynamics in Dhaka, Bangladesh: Trends, Drivers, and Projections. *[Journal]*, [Year].

---

## License

Data sourced from the Bangladesh Meteorological Department and Global Forest Watch. Analysis code is available for academic use.
