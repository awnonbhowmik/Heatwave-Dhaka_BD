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
│   ├── fig1_study_area.png            # Study area: Dhaka City District, Bangladesh
│   ├── fig2_overview.png              # Warming trend, heatwave days, deforestation
│   ├── fig3_temperature_structure.png # Daily range, seasonal cycle, decadal distribution
│   ├── fig4_heatwave_characteristics.png  # Annual counts, monthly distribution, event durations
│   ├── fig5_heatwave_calendar.png     # Year–month heatmap across all 52 years
│   ├── fig6_heat_index.png            # NWS apparent temperature burden and trend
│   ├── fig7_recovery_gap.png          # Nighttime recovery gap — asymmetric Tmax/Tmin warming
│   ├── fig8_compound_events.png       # Compound hot+dry events
│   ├── fig9_correlation_matrix.png    # Pearson correlation matrix of climate drivers
│   ├── fig10_forecast_combined.png    # Combined 3×2 forecast: SARIMA · ARIMA · LSTM
│   ├── fig11_model_comparison.png     # Multi-model summary: trajectories · AIC · ML metrics
│   ├── supp_rf_forecast.png           # Supplementary: Random Forest (notebook only)
│   └── supp_xgboost_forecast.png      # Supplementary: XGBoost (notebook only)
├── analysis.ipynb                     # Full reproducible pipeline (see below)
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

Figures are numbered in manuscript order. All are saved to `figures/` at 300 DPI.

| # | File | Panels | Description |
|---|------|--------|-------------|
| 1 | `fig1_study_area.png` | 1 | Study area: Dhaka City District, Bangladesh — location map with district boundary |
| 2 | `fig2_overview.png` | A–C | $T_\mathrm{max}$ warming trend · annual heatwave days (1972–2024) · deforestation overlay |
| 3 | `fig3_temperature_structure.png` | A–D | Daily temperature range (2020–2024) · annual $T_\mathrm{max}$ trend · seasonal climatology · decadal boxplots |
| 4 | `fig4_heatwave_characteristics.png` | A–C | Annual heatwave day counts · monthly distribution (Apr–May peak) · event duration histogram |
| 5 | `fig5_heatwave_calendar.png` | 1 | Year–month heatmap across all 52 years — April–May concentration visible throughout |
| 6 | `fig6_heat_index.png` | A–C | NWS Heat Index vs $T_\mathrm{max}$ trend · humidity heat burden on heatwave days · extreme HI exceedance days |
| 7 | `fig7_recovery_gap.png` | A–C | Nighttime recovery gap ($T_\mathrm{max}-T_\mathrm{min}$) · $T_\mathrm{min}$ trend · asymmetric warming rates (UHI indicator) |
| 8 | `fig8_compound_events.png` | A–C | Compound hot+dry event frequency · correlation with heatwave days · seasonal distribution |
| 9 | `fig9_correlation_matrix.png` | 1 | Pearson $r$ matrix across 10 climate drivers — $T_\mathrm{max}$, $T_\mathrm{min}$, HI, VPD, soil moisture, etc. |
| 10 | `fig10_forecast_combined.png` | A–F | **Combined 3×2 forecast panel** — one row per model (SARIMA · ARIMA · LSTM); left column: monthly $T_\mathrm{max}$ forecast with 95% CI; right column: projected heatwave days 2025–2029 |
| 11 | `fig11_model_comparison.png` | A–D | Multi-model summary — (A) annual $T_\mathrm{max}$ trajectories for all 5 models · (B) projected heatwave days · (C) AIC comparison (SARIMA best fit) · (D) ML test metrics (RMSE / $R^2$) |
| — | `supp_rf_forecast.png` | A–B | Supplementary: Random Forest forecast (notebook only — insufficient annual training data for manuscript) |
| — | `supp_xgboost_forecast.png` | A–B | Supplementary: XGBoost forecast (notebook only — same limitation as RF) |

---

## Reproducing the analysis

### Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

### Setup

```bash
conda create -n heatwave python=3.13
conda activate heatwave
pip install \
    numpy==2.3.1 \
    pandas==2.3.1 \
    matplotlib==3.10.0 \
    seaborn==0.13.2 \
    scipy==1.16.0 \
    statsmodels==0.14.5 \
    scikit-learn==1.7.1 \
    xgboost==3.2.0 \
    torch==2.8.0 \
    jupyter ipykernel
python -m ipykernel install --user --name heatwave
```

> **Note:** `torch==2.8.0` installs the CPU build by default. For GPU support, follow the [PyTorch install guide](https://pytorch.org/get-started/locally/) to select the appropriate CUDA version. The notebook forces CPU execution (`DEVICE = "cpu"`) so GPU is not required.

### Run

```bash
cd Heatwave-Dhaka_BD
jupyter notebook analysis.ipynb
```

Select kernel **heatwave** and run all cells (Kernel → Restart & Run All). Execution takes ~5–8 minutes (LSTM training on CPU).

All figures are saved to `figures/` at 300 DPI automatically.

---

## Models used

| Model | Order / Config | AIC | Test $R^2$ | Notes |
|-------|---------------|-----|-----------|-------|
| SARIMA | (3,1,0)×(1,0,0)₁₂ | 2109.7 | — | **Primary forecast** — best AIC, captures monsoon seasonality |
| ARIMA | (5,1,0)×(1,0,0)₁₂ | 2159.1 | — | Comparison baseline; higher AIC |
| LSTM | 2-layer, 64 hidden, SEQ=24 | — | 0.81 | Strong 1-step fit; iterative multi-step forecast reverts to trend mean |
| Random Forest | 500 trees, depth=4 | — | −0.15 | Supplementary only — insufficient annual training data |
| XGBoost | 300 estimators, depth=3 | — | −0.24 | Supplementary only — same limitation as RF |

SARIMA and ARIMA individual forecasts with 95% CI, alongside LSTM, are shown together in **fig10**. All five models are compared in **fig11**.

---

## Citation

> [Authors]. Heatwave Dynamics in Dhaka, Bangladesh: Trends, Drivers, and Projections. *[Journal]*, [Year].

---

## License

Data sourced from the Bangladesh Meteorological Department and Global Forest Watch. Analysis code is available for academic use.
