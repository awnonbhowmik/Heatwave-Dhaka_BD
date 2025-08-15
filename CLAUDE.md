# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a climate research project analyzing heatwave trends in Dhaka, Bangladesh from 1972-2024. The repository contains comprehensive analysis of temperature patterns, deforestation impacts, and predictive modeling for future climate scenarios.

**Key Research Focus:**
- Historical heatwave pattern analysis (1972-2024)
- Climate-deforestation correlation studies
- Advanced time series forecasting (ARIMA, SARIMA, LSTM, ML models)
- Statistical trend analysis and climate risk assessment

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (if exists)
source .venv/bin/activate

# Install dependencies (if requirements.txt exists)
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima tensorflow xgboost

# For Jupyter notebooks
pip install jupyter
```

### Running Analysis
```bash
# Launch Jupyter for interactive analysis
jupyter notebook

# Run modular analysis (recommended)
jupyter notebook main_modular.ipynb

# Run complete analysis
jupyter notebook main.ipynb
```

### Python Execution
```bash
# Run individual modules
python -c "from src import data_loader; data, threshold = data_loader.load_heatwave_data()"
python -c "from src import predictive_models; # Run specific model functions"
```

## Code Architecture

### Modular Structure
The codebase follows a modular design with specialized modules:

**Core Analysis Modules (`src/`):**
- `data_loader.py` - Data loading, preprocessing, and feature engineering
- `statistical_analysis.py` - Comprehensive statistical analysis and trend detection
- `visualization.py` - Plotting functions for all chart types and dashboards
- `predictive_models.py` - Advanced forecasting models (ARIMA, SARIMA, LSTM, ML)

**Main Analysis Files:**
- `main_modular.ipynb` - **Recommended**: Clean, modular analysis workflow
- `main.ipynb` - Complete analysis in single notebook

### Data Architecture
**Input Data:**
- `data/1972_2024_Heatwave_Daily.xlsx` - Daily climate records (52 years)
- `data/GFW_Dhaka.csv` - Deforestation data from Global Forest Watch

**Key Data Processing:**
- Heatwave threshold: 36°C (configurable in `data_loader.py:21`)
- Time series features: seasonal, cyclical encoding, lagged features
- Combined climate-deforestation datasets for correlation analysis

### Predictive Modeling Architecture

**Model Types Implemented:**
1. **ARIMA** - Basic time series forecasting
2. **SARIMA** - Seasonal time series with enhanced decomposition
3. **LSTM** - Deep learning sequences (60-day windows)
4. **Random Forest** - Feature-based ML regression
5. **XGBoost** - Gradient boosting with feature importance

**Model Workflow:**
- Feature preparation and scaling in `HeatwavePredictor` class
- Cross-validation with `TimeSeriesSplit` for temporal data
- 5-year forecasting (2025-2029) with confidence intervals
- Comprehensive performance evaluation and visualization

### Analysis Capabilities

**Statistical Analysis:**
- Temperature trend analysis with linear regression
- Stationarity testing (ADF test)
- Deforestation-temperature correlation (Pearson/Spearman)
- Period comparison (pre/post 2000)
- Distribution analysis and normality testing

**Visualization Suite:**
- Temperature trends and seasonal patterns
- Heatwave frequency and duration analysis
- Deforestation impact visualization
- Correlation matrices for climate variables
- Model prediction comparisons and confidence intervals
- Comprehensive summary dashboards

## Key Development Patterns

### Module Import Pattern
```python
# Recommended import style used in main_modular.ipynb
import data_loader
import statistical_analysis  
import visualization
import predictive_models

# Reload for development
import importlib
importlib.reload(module_name)
```

### Data Flow Pattern
```python
# Standard workflow
data, threshold = data_loader.load_heatwave_data()
deforestation_data, tree_loss_by_year = data_loader.load_deforestation_data()
combined_data, annual_temp_stats = data_loader.combine_datasets(data, tree_loss_by_year)

# Analysis
statistical_results = statistical_analysis.comprehensive_statistical_analysis(...)
visualization.plot_temperature_trends(...)

# Predictive modeling
predictor = predictive_models.HeatwavePredictor(data, tree_loss_by_year)
predictor.prepare_features()
predictor.fit_arima_model()  # or other models
```

### Model Performance Tracking
All models store results in standardized format:
- Training/test metrics (RMSE, R², MAE)
- Feature importance rankings
- Future forecasts with annual breakdowns
- Model configuration and parameters

## Dependencies and Environment

**Core Scientific Stack:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Machine learning utilities and metrics

**Time Series and Advanced Models:**
- `statsmodels` - ARIMA, SARIMA, statistical testing
- `pmdarima` - Auto-ARIMA parameter selection (optional)
- `tensorflow` - LSTM deep learning models (optional)
- `xgboost` - Gradient boosting (optional)

**Note:** Advanced models (TensorFlow, XGBoost, pmdarima) gracefully handle missing dependencies with fallback messages.

## Climate Data Context

**Temperature Thresholds:**
- Heatwave definition: >36°C daily maximum
- Historical average: ~27°C (varies seasonally)
- Data quality: 52 years of continuous daily records

**Seasonal Patterns:**
- Peak heat: April-May (pre-monsoon)
- Monsoon cooling: June-September  
- Winter minimum: December-February
- Cyclical encoding captures annual patterns

**Research Implications:**
- Focus on trend analysis over absolute prediction accuracy
- Climate models emphasize long-term patterns over daily precision
- Statistical significance testing for all trend claims
- Confidence intervals essential for future projections