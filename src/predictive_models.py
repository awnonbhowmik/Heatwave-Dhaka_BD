"""
Enhanced Predictive Modeling Module for Climate Analysis

Provides state-of-the-art climate forecasting with comprehensive modeling approaches,
advanced analytics, uncertainty quantification, and robust evaluation frameworks.
"""

import logging
import typing
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallbacks
OPTIONAL_DEPS = {}

# TensorFlow/Keras for deep learning
try:
    import tensorflow as tf  # type: ignore

    # Help type checker accept tf.keras accesses
    tf = typing.cast(Any, tf)
    OPTIONAL_DEPS["tensorflow"] = True
    logger.info("✓ TensorFlow available for deep learning models")
except ImportError:
    OPTIONAL_DEPS["tensorflow"] = False
    # Provide a harmless placeholder to satisfy type checkers
    import types as _types

    tf = typing.cast(Any, _types.SimpleNamespace())
    logger.warning("TensorFlow not available. Deep learning models will be disabled.")

# XGBoost for gradient boosting
try:
    import xgboost as xgb

    OPTIONAL_DEPS["xgboost"] = True
    logger.info("✓ XGBoost available for gradient boosting")
except ImportError:
    OPTIONAL_DEPS["xgboost"] = False
    logger.warning("XGBoost not available. XGBoost models will be disabled.")

# pmdarima for auto-ARIMA
try:
    from pmdarima import auto_arima  # type: ignore

    OPTIONAL_DEPS["pmdarima"] = True
    logger.info("✓ pmdarima available for auto-ARIMA")
except ImportError:
    OPTIONAL_DEPS["pmdarima"] = False
    logger.warning("pmdarima not available. Auto-ARIMA will use default parameters.")
    auto_arima = None

# Bayesian optimization for hyperparameter tuning
try:
    from skopt import BayesSearchCV  # type: ignore
    from skopt.space import Integer, Real  # type: ignore

    OPTIONAL_DEPS["scikit_optimize"] = True
    logger.info("✓ scikit-optimize available for Bayesian optimization")
except ImportError:
    OPTIONAL_DEPS["scikit_optimize"] = False
    logger.warning(
        "scikit-optimize not available. Using GridSearchCV for hyperparameter tuning."
    )
    BayesSearchCV = None
    Integer = Real = None

# Global constants
DEFAULT_FIGSIZE = (16, 12)
DPI = 300
FONT_SIZES = {"title": 16, "subtitle": 14, "label": 12, "tick": 10, "legend": 11}
TEMP_COLUMN = "Dhaka Temperature [2 m elevation corrected]"
FORECAST_YEARS = 5
CONFIDENCE_LEVEL = 0.95


# Helper: resolve temperature min/max column names robustly across schemas
def _resolve_temp_min_max_generic(
    df: pd.DataFrame, base_col: str
) -> Tuple[Optional[str], Optional[str]]:
    """Return (min_col, max_col) if detectable, else (None, None).

    Tries common suffix patterns and infers using sample means when both .1/.2 exist.
    """
    # Prefer explicit names if present
    candidates_max = [
        f"{base_col}.max",
        f"{base_col}.1",
        f"{base_col}.2",
        f"{base_col} Max",
        f"{base_col} Maximum",
    ]
    candidates_min = [
        f"{base_col}.min",
        f"{base_col}.2",
        f"{base_col}.1",
        f"{base_col} Min",
        f"{base_col} Minimum",
    ]

    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col1, col2 = f"{base_col}.1", f"{base_col}.2"
    inferred_max = inferred_min = None
    if col1 in df.columns and col2 in df.columns:
        sample = df[[col1, col2]].dropna().head(100)
        if not sample.empty:
            if sample[col1].mean() >= sample[col2].mean():
                inferred_max, inferred_min = col1, col2
            else:
                inferred_max, inferred_min = col2, col1

    max_col = inferred_max or pick(candidates_max)
    min_col = inferred_min or pick(candidates_min)

    if max_col == min_col:
        return None, None
    return min_col, max_col


class ClimateMetrics:
    """Climate-specific evaluation metrics for model assessment"""

    @staticmethod
    def climate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error optimized for climate data"""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def climate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error for temperature predictions"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def climate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error for climate forecasting"""
        # Avoid division by zero for temperature data
        mask = np.abs(y_true) > 0.1  # Temperature rarely near 0°C
        if mask.sum() == 0:
            return np.inf
        return mean_absolute_percentage_error(y_true[mask], y_pred[mask])

    @staticmethod
    def seasonal_bias(
        y_true: np.ndarray, y_pred: np.ndarray, season_idx: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate seasonal bias in predictions"""
        seasonal_bias = {}
        for season in np.unique(season_idx):
            mask = season_idx == season
            if mask.sum() > 0:
                bias = np.mean(y_pred[mask] - y_true[mask])
                seasonal_bias[f"season_{season}_bias"] = bias
        return seasonal_bias

    @staticmethod
    def extreme_event_skill(
        y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 36.0
    ) -> Dict[str, float]:
        """Evaluate model skill for extreme temperature events (heatwaves)"""
        true_extreme = y_true > threshold
        pred_extreme = y_pred > threshold

        # Confusion matrix elements
        tp = np.sum(true_extreme & pred_extreme)  # True positives
        fp = np.sum(~true_extreme & pred_extreme)  # False positives
        fn = np.sum(true_extreme & ~pred_extreme)  # False negatives
        tn = np.sum(~true_extreme & ~pred_extreme)  # True negatives

        # Calculate skill metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0

        return {
            "extreme_precision": float(precision),
            "extreme_recall": float(recall),
            "extreme_f1_score": float(f1_score),
            "extreme_accuracy": float(accuracy),
            "extreme_true_positives": float(tp),
            "extreme_false_positives": float(fp),
            "extreme_false_negatives": float(fn),
        }

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Measure accuracy of predicted trend direction"""
        if len(y_true) < 2:
            return 0.0

        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0

        return np.mean(true_direction == pred_direction)

    @staticmethod
    def comprehensive_evaluation(
        y_true: np.ndarray, y_pred: np.ndarray, season_idx: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Comprehensive climate model evaluation"""
        metrics = {
            "mae": ClimateMetrics.climate_mae(y_true, y_pred),
            "rmse": ClimateMetrics.climate_rmse(y_true, y_pred),
            "mape": ClimateMetrics.climate_mape(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "directional_accuracy": ClimateMetrics.directional_accuracy(y_true, y_pred),
        }

        # Add extreme event metrics
        extreme_metrics = ClimateMetrics.extreme_event_skill(y_true, y_pred)
        metrics.update(extreme_metrics)

        # Add seasonal bias if season information available
        if season_idx is not None:
            seasonal_metrics = ClimateMetrics.seasonal_bias(y_true, y_pred, season_idx)
            metrics.update(seasonal_metrics)

        return metrics


class AdvancedFeatureEngineering:
    """Advanced feature engineering for climate time series"""

    @staticmethod
    def create_temporal_features(
        data: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """Create comprehensive temporal features"""
        data_copy = data.copy()

        # Basic temporal features
        data_copy["Year"] = data_copy[timestamp_col].dt.year
        data_copy["Month"] = data_copy[timestamp_col].dt.month
        data_copy["Day"] = data_copy[timestamp_col].dt.day
        data_copy["DayOfYear"] = data_copy[timestamp_col].dt.dayofyear
        data_copy["WeekOfYear"] = data_copy[timestamp_col].dt.isocalendar().week
        data_copy["DayOfWeek"] = data_copy[timestamp_col].dt.dayofweek
        data_copy["Quarter"] = data_copy[timestamp_col].dt.quarter
        data_copy["Season"] = data_copy["Month"] % 12 // 3 + 1

        # Cyclical encoding for periodic features
        data_copy["Month_sin"] = np.sin(2 * np.pi * data_copy["Month"] / 12)
        data_copy["Month_cos"] = np.cos(2 * np.pi * data_copy["Month"] / 12)
        data_copy["DayOfYear_sin"] = np.sin(2 * np.pi * data_copy["DayOfYear"] / 365.25)
        data_copy["DayOfYear_cos"] = np.cos(2 * np.pi * data_copy["DayOfYear"] / 365.25)
        data_copy["DayOfWeek_sin"] = np.sin(2 * np.pi * data_copy["DayOfWeek"] / 7)
        data_copy["DayOfWeek_cos"] = np.cos(2 * np.pi * data_copy["DayOfWeek"] / 7)

        # Time-based indicators
        data_copy["IsWeekend"] = data_copy["DayOfWeek"].isin([5, 6]).astype(int)
        data_copy["IsMonthStart"] = data_copy[timestamp_col].dt.is_month_start.astype(
            int
        )
        data_copy["IsMonthEnd"] = data_copy[timestamp_col].dt.is_month_end.astype(int)
        data_copy["IsYearStart"] = data_copy[timestamp_col].dt.is_year_start.astype(int)
        data_copy["IsYearEnd"] = data_copy[timestamp_col].dt.is_year_end.astype(int)

        # Climate season indicators (Bangladesh-specific)
        data_copy["IsMonsoon"] = data_copy["Month"].isin([6, 7, 8, 9]).astype(int)
        data_copy["IsPreMonsoon"] = data_copy["Month"].isin([3, 4, 5]).astype(int)
        data_copy["IsPostMonsoon"] = data_copy["Month"].isin([10, 11]).astype(int)
        data_copy["IsWinter"] = data_copy["Month"].isin([12, 1, 2]).astype(int)

        # Days since epoch for trend modeling
        epoch = data_copy[timestamp_col].min()
        data_copy["Days_Since_Epoch"] = (data_copy[timestamp_col] - epoch).dt.days
        data_copy["Years_Since_Epoch"] = data_copy["Days_Since_Epoch"] / 365.25

        return data_copy

    @staticmethod
    def create_lagged_features(
        data: pd.DataFrame, target_col: str, lags: List[int] = [1, 7, 14, 30, 90, 365]
    ) -> pd.DataFrame:
        """Create sophisticated lagged features"""
        data_copy = data.copy()

        for lag in lags:
            # Direct lag
            data_copy[f"{target_col}_lag_{lag}"] = data_copy[target_col].shift(lag)

            # Lag differences
            data_copy[f"{target_col}_lag_{lag}_diff"] = (
                data_copy[target_col] - data_copy[f"{target_col}_lag_{lag}"]
            )

            # Lag ratios
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data_copy[f"{target_col}_lag_{lag}_ratio"] = (
                    data_copy[target_col] / data_copy[f"{target_col}_lag_{lag}"]
                )

        return data_copy

    @staticmethod
    def create_rolling_features(
        data: pd.DataFrame, target_col: str, windows: List[int] = [7, 14, 30, 90, 365]
    ) -> pd.DataFrame:
        """Create comprehensive rolling window features"""
        data_copy = data.copy()

        for window in windows:
            # Rolling statistics
            rolling = data_copy[target_col].rolling(window=window, center=True)
            data_copy[f"{target_col}_roll_{window}_mean"] = rolling.mean()
            data_copy[f"{target_col}_roll_{window}_std"] = rolling.std()
            data_copy[f"{target_col}_roll_{window}_min"] = rolling.min()
            data_copy[f"{target_col}_roll_{window}_max"] = rolling.max()
            data_copy[f"{target_col}_roll_{window}_median"] = rolling.median()

            # Rolling percentiles
            data_copy[f"{target_col}_roll_{window}_q25"] = rolling.quantile(0.25)
            data_copy[f"{target_col}_roll_{window}_q75"] = rolling.quantile(0.75)

            # Anomaly indicators
            mean_col = f"{target_col}_roll_{window}_mean"
            std_col = f"{target_col}_roll_{window}_std"
            data_copy[f"{target_col}_roll_{window}_zscore"] = (
                data_copy[target_col] - data_copy[mean_col]
            ) / data_copy[std_col]

            # Range indicators
            data_copy[f"{target_col}_roll_{window}_range"] = (
                data_copy[f"{target_col}_roll_{window}_max"]
                - data_copy[f"{target_col}_roll_{window}_min"]
            )

        return data_copy

    @staticmethod
    def create_climate_indices(data: pd.DataFrame) -> pd.DataFrame:
        """Create climate-specific indices and indicators"""
        data_copy = data.copy()

        # Enhanced heat index calculation
        if all(
            col in data.columns
            for col in [TEMP_COLUMN, "Dhaka Relative Humidity [2 m]"]
        ):
            temp_f = data_copy[TEMP_COLUMN] * 9 / 5 + 32  # Convert to Fahrenheit
            humidity = data_copy["Dhaka Relative Humidity [2 m]"]

            # Steadman's heat index formula (simplified version)
            heat_index = (
                -42.379
                + 2.04901523 * temp_f
                + 10.14333127 * humidity
                - 0.22475541 * temp_f * humidity
                - 6.83783e-3 * temp_f**2
                - 5.481717e-2 * humidity**2
                + 1.22874e-3 * temp_f**2 * humidity
                + 8.5282e-4 * temp_f * humidity**2
                - 1.99e-6 * temp_f**2 * humidity**2
            )

            # Convert back to Celsius
            data_copy["Heat_Index"] = (heat_index - 32) * 5 / 9

            # Heat stress categories
            data_copy["Heat_Stress_Level"] = pd.cut(
                data_copy["Heat_Index"],
                bins=[-np.inf, 27, 32, 41, 54, np.inf],
                labels=[0, 1, 2, 3, 4],  # 0=Caution, 4=Extreme Danger
            ).astype(float)

        # Drought indicators
        if "Dhaka Precipitation Total" in data.columns:
            precip_col = "Dhaka Precipitation Total"
            # 30-day and 90-day precipitation deficits
            for window in [30, 90]:
                precip_sum = data_copy[precip_col].rolling(window=window).sum()
                precip_mean = data_copy[precip_col].rolling(window=365).mean() * window
                data_copy[f"Precip_Deficit_{window}d"] = precip_mean - precip_sum
                data_copy[f"Drought_Index_{window}d"] = (
                    data_copy[f"Precip_Deficit_{window}d"] / precip_mean
                ).clip(0, 1)

        # Temperature amplitude and variability
        if TEMP_COLUMN in data.columns:
            # Daily temperature range if min/max available
            temp_min_col, temp_max_col = _resolve_temp_min_max_generic(
                data_copy, TEMP_COLUMN
            )

            if temp_max_col and temp_min_col:
                data_copy["Daily_Temp_Range"] = (
                    data_copy[temp_max_col] - data_copy[temp_min_col]
                )
                data_copy["Temp_Range_Percentile"] = data_copy["Daily_Temp_Range"].rank(
                    pct=True
                )

        # Seasonal temperature anomalies
        monthly_normals = data_copy.groupby("Month")[TEMP_COLUMN].transform("mean")
        data_copy["Monthly_Temp_Anomaly"] = data_copy[TEMP_COLUMN] - monthly_normals

        seasonal_normals = data_copy.groupby("Season")[TEMP_COLUMN].transform("mean")
        data_copy["Seasonal_Temp_Anomaly"] = data_copy[TEMP_COLUMN] - seasonal_normals

        return data_copy


class EnhancedClimatePredictor:
    """Enhanced climate prediction system with comprehensive modeling approaches"""

    def __init__(self, data: pd.DataFrame, tree_loss_by_year: pd.DataFrame):
        """
        Initialize the enhanced climate predictor

        Args:
            data: Daily climate data
            tree_loss_by_year: Annual deforestation data
        """
        self.data = data.copy()
        self.tree_loss_by_year = tree_loss_by_year.copy()
        self.models = {}
        self.forecasts = {}
        self.model_performance = {}
        self.feature_data = None
        self.scalers = {}
        self.consolidated_forecasts = {}

        logger.info(
            f"Initialized EnhancedClimatePredictor with {len(data)} observations"
        )
        logger.info(
            f"Time range: {data['timestamp'].min()} to {data['timestamp'].max()}"
        )

    def prepare_enhanced_features(self) -> pd.DataFrame:
        """Prepare comprehensive feature set for modeling"""
        logger.info("🔧 Preparing enhanced features for climate modeling...")

        # Start with temporal features
        feature_data = AdvancedFeatureEngineering.create_temporal_features(self.data)

        # Add lagged features
        feature_data = AdvancedFeatureEngineering.create_lagged_features(
            feature_data, TEMP_COLUMN
        )

        # Add rolling features
        feature_data = AdvancedFeatureEngineering.create_rolling_features(
            feature_data, TEMP_COLUMN
        )

        # Add climate indices
        feature_data = AdvancedFeatureEngineering.create_climate_indices(feature_data)

        # Merge with deforestation data
        if not self.tree_loss_by_year.empty:
            deforest_cols = ["Year"]
            if "Total_Loss_ha" in self.tree_loss_by_year.columns:
                deforest_cols.append("Total_Loss_ha")
            elif "umd_tree_cover_loss__ha" in self.tree_loss_by_year.columns:
                deforest_cols.append("umd_tree_cover_loss__ha")

            feature_data = feature_data.merge(
                self.tree_loss_by_year[deforest_cols], on="Year", how="left"
            )

            # Fill missing deforestation values and create cumulative
            deforest_col = deforest_cols[1] if len(deforest_cols) > 1 else None
            if deforest_col:
                feature_data[deforest_col] = feature_data[deforest_col].fillna(0)
                feature_data["Cumulative_Deforestation"] = feature_data.groupby("Year")[
                    deforest_col
                ].cumsum()

        # Add interaction features for key climate variables
        if all(
            col in feature_data.columns
            for col in [TEMP_COLUMN, "Dhaka Relative Humidity [2 m]"]
        ):
            feature_data["Temp_Humidity_Interaction"] = (
                feature_data[TEMP_COLUMN]
                * feature_data["Dhaka Relative Humidity [2 m]"]
            )

        if all(col in feature_data.columns for col in ["Month_sin", "Month_cos"]):
            feature_data["Seasonal_Cycle_Intensity"] = np.sqrt(
                feature_data["Month_sin"] ** 2 + feature_data["Month_cos"] ** 2
            )

        self.feature_data = feature_data

        # Log feature engineering results
        original_cols = len(self.data.columns)
        new_cols = len(feature_data.columns)
        logger.info("✅ Feature engineering completed:")
        logger.info(f"   Original features: {original_cols}")
        logger.info(f"   Enhanced features: {new_cols}")
        logger.info(f"   Added features: {new_cols - original_cols}")

        return feature_data

    def prepare_model_data(
        self,
        target_col: str = TEMP_COLUMN,
        feature_selection_method: str = "correlation",
        max_features: int = 50,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling with feature selection"""
        if self.feature_data is None:
            _ = self.prepare_enhanced_features()
        assert (
            self.feature_data is not None
        ), "Feature engineering failed to produce data"

        # Get available features (exclude target and non-predictive columns)
        exclude_cols = [
            target_col,
            "timestamp",
            "Year",
            "Month",
            "Day",  # Target and basic time
            "DayOfYear",
            "WeekOfYear",
            "Quarter",  # Redundant with cyclical encoding
        ]

        # Get numeric columns only
        numeric_cols = self.feature_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Remove features with too many missing values
        missing_threshold = 0.3
        feature_cols = [
            col
            for col in feature_cols
            if self.feature_data[col].isna().mean() < missing_threshold
        ]

        logger.info(f"Available features for modeling: {len(feature_cols)}")

        # Feature selection
        if (
            feature_selection_method == "correlation"
            and len(feature_cols) > max_features
        ):
            # Select features based on correlation with target
            correlations = (
                self.feature_data[feature_cols]
                .corrwith(self.feature_data[target_col])
                .abs()
                .sort_values(ascending=False)
            )

            selected_features = correlations.head(max_features).index.tolist()
            logger.info(
                f"Selected {len(selected_features)} features using correlation method"
            )
        else:
            selected_features = feature_cols[:max_features]

        # Prepare final datasets
        model_data = self.feature_data[selected_features + [target_col]].copy()

        # Replace inf/-inf from ratio features before dropping NaNs
        model_data = model_data.replace([np.inf, -np.inf], np.nan)

        # Handle missing values
        model_data = model_data.dropna()

        X = model_data[selected_features]
        y = model_data[target_col]

        logger.info(
            f"Model data prepared: {len(X)} samples, {len(selected_features)} features"
        )

        return X, y

    def fit_enhanced_arima_sarima(
        self, seasonal: bool = True, auto_params: bool = True
    ):
        """Enhanced ARIMA/SARIMA modeling with advanced parameter optimization"""
        model_type = "SARIMA" if seasonal else "ARIMA"
        logger.info(f"🔮 Enhanced {model_type} Time Series Forecasting")
        logger.info("=" * 60)

        try:
            # Prepare monthly temperature data
            data_copy = self.data.copy()
            data_copy["YearMonth"] = data_copy["timestamp"].dt.to_period("M")
            monthly_temp = data_copy.groupby("YearMonth")[TEMP_COLUMN].mean()
            # Robust index conversion to datetime for forecasting
            try:
                if hasattr(monthly_temp.index, "to_timestamp"):
                    monthly_temp.index = monthly_temp.index.to_timestamp()  # type: ignore[attr-defined]
                else:
                    monthly_temp.index = pd.to_datetime(monthly_temp.index.astype(str))
            except Exception:
                monthly_temp.index = pd.to_datetime(monthly_temp.index.astype(str))

            logger.info(f"Time series shape: {monthly_temp.shape}")
            logger.info(
                f"Date range: {monthly_temp.index.min()} to {monthly_temp.index.max()}"
            )

            # 1. Enhanced stationarity testing
            logger.info("\n1. STATIONARITY ANALYSIS")
            stationarity_results = self._test_stationarity(monthly_temp)

            # 2. Seasonal decomposition
            logger.info("\n2. SEASONAL DECOMPOSITION")
            decomposition = seasonal_decompose(
                monthly_temp, model="additive", period=12
            )

            # 3. Model parameter optimization
            logger.info("\n3. MODEL PARAMETER OPTIMIZATION")
            if auto_params and OPTIONAL_DEPS["pmdarima"] and auto_arima is not None:
                if seasonal:
                    model = auto_arima(
                        monthly_temp,
                        seasonal=True,
                        m=12,
                        max_p=3,
                        max_q=3,
                        max_P=2,
                        max_Q=2,
                        max_d=2,
                        max_D=1,
                        suppress_warnings=True,
                        stepwise=True,
                        trace=True,
                        information_criterion="aic",
                    )
                    order = model.order
                    seasonal_order = model.seasonal_order
                else:
                    model = auto_arima(
                        monthly_temp,
                        seasonal=False,
                        max_p=5,
                        max_q=5,
                        max_d=2,
                        suppress_warnings=True,
                        stepwise=True,
                        trace=True,
                    )
                    order = model.order
                    seasonal_order = None
            else:
                # Default parameters
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 12) if seasonal else None

            logger.info(f"Selected model order: {order}")
            if seasonal_order:
                logger.info(f"Seasonal order: {seasonal_order}")

            # 4. Fit the model
            logger.info("\n4. MODEL FITTING")
            if seasonal:
                fitted_model = SARIMAX(
                    monthly_temp,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            else:
                fitted_model = ARIMA(monthly_temp, order=order).fit()

            # 5. Model diagnostics
            logger.info("\n5. MODEL DIAGNOSTICS")
            diagnostics = self._model_diagnostics(fitted_model, monthly_temp)

            # 6. Generate forecasts
            logger.info("\n6. GENERATING FORECASTS")
            forecast_steps = 60  # 5 years
            forecast_result = fitted_model.get_forecast(steps=forecast_steps)  # type: ignore[attr-defined]
            forecast = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()

            # Create future dates
            last_date = monthly_temp.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),  # type: ignore[operator]
                periods=forecast_steps,
                freq="M",
            )

            # 7. Calculate forecast statistics
            historical_avg = monthly_temp.mean()
            forecast_avg = forecast.mean()

            # Annual forecasts
            annual_forecasts = []
            for year in range(FORECAST_YEARS):
                year_start = year * 12
                year_end = (year + 1) * 12
                if year_end <= len(forecast):
                    annual_avg = forecast[year_start:year_end].mean()
                    annual_forecasts.append(annual_avg)

            # 8. Store comprehensive results
            model_key = "sarima" if seasonal else "arima"

            self.models[model_key] = fitted_model
            self.forecasts[model_key] = {
                "forecast": forecast,
                "forecast_ci": forecast_ci,
                "future_dates": future_dates,
                "historical_data": monthly_temp,
                "decomposition": decomposition,
                "order": order,
                "seasonal_order": seasonal_order,
                "annual_forecasts": annual_forecasts,
                "stationarity_results": stationarity_results,
                "diagnostics": diagnostics,
                "model_summary": {
                    "aic": fitted_model.aic,  # type: ignore[attr-defined]
                    "bic": fitted_model.bic,  # type: ignore[attr-defined]
                    "llf": fitted_model.llf,  # type: ignore[attr-defined]
                    "historical_avg": historical_avg,
                    "forecast_avg": forecast_avg,
                    "forecast_increase": forecast_avg - historical_avg,
                    "seasonal_component": seasonal,
                },
            }

            # Calculate performance metrics on in-sample data
            try:
                residuals = fitted_model.resid  # type: ignore[attr-defined]
            except Exception:
                try:
                    residuals = monthly_temp - fitted_model.fittedvalues  # type: ignore[attr-defined]
                except Exception:
                    residuals = pd.Series(dtype=float)

            self.model_performance[model_key] = {
                "in_sample_mae": np.mean(np.abs(residuals)),
                "in_sample_rmse": np.sqrt(np.mean(residuals**2)),
                "in_sample_mape": np.mean(np.abs(residuals / monthly_temp)) * 100,
                "ljung_box_pvalue": diagnostics.get("ljung_box_pvalue", np.nan),
                "jarque_bera_pvalue": diagnostics.get("jarque_bera_pvalue", np.nan),
            }

            logger.info(f"✅ {model_type} modeling completed successfully!")
            logger.info(f"📊 Model AIC: {fitted_model.aic:.2f}")  # type: ignore[attr-defined]
            logger.info(f"🌡️ Forecast increase: {forecast_avg - historical_avg:.3f}°C")

        except Exception as e:
            logger.error(f"❌ {model_type} modeling failed: {e}")
            import traceback

            traceback.print_exc()

    def fit_enhanced_lstm(
        self,
        sequence_length: int = 60,
        architecture: str = "advanced",
        epochs: int = 100,
    ):
        """Enhanced LSTM with attention mechanisms and advanced architectures"""
        if not OPTIONAL_DEPS["tensorflow"]:
            logger.warning("TensorFlow not available. Skipping LSTM modeling.")
            return

        logger.info("🧠 Enhanced LSTM Deep Learning Forecasting")
        logger.info("=" * 60)

        try:
            # Prepare features for LSTM
            if self.feature_data is None:
                _ = self.prepare_enhanced_features()
            assert (
                self.feature_data is not None
            ), "Feature engineering failed to produce data"

            # Select features for LSTM
            lstm_features = [
                TEMP_COLUMN,
                "Dhaka Relative Humidity [2 m]",
                "Dhaka Precipitation Total",
                "Month_sin",
                "Month_cos",
                "DayOfYear_sin",
                "DayOfYear_cos",
                "Heat_Index",
            ]

            # Add deforestation if available
            if "Total_Loss_ha" in self.feature_data.columns:
                lstm_features.append("Total_Loss_ha")
            elif "umd_tree_cover_loss__ha" in self.feature_data.columns:
                lstm_features.append("umd_tree_cover_loss__ha")

            # Filter available features
            available_features = [
                f for f in lstm_features if f in self.feature_data.columns
            ]
            target_col = TEMP_COLUMN

            # Prepare data
            lstm_data = self.feature_data[available_features].dropna()
            logger.info(f"LSTM data shape: {lstm_data.shape}")
            logger.info(f"Features: {available_features}")

            # Scale data
            feature_scaler = RobustScaler()
            target_scaler = RobustScaler()

            # Include target as part of input features for closed-loop forecasting
            feature_cols = [f for f in available_features if f != target_col]
            feature_cols_with_target = [target_col] + feature_cols

            scaled_features = feature_scaler.fit_transform(
                lstm_data[feature_cols_with_target]
            )
            scaled_target = target_scaler.fit_transform(lstm_data[[target_col]])

            # Create sequences
            X, y = self._create_sequences(
                scaled_features, scaled_target.flatten(), sequence_length
            )

            logger.info(f"Sequence data shape: X={X.shape}, y={y.shape}")

            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Build enhanced LSTM model
            logger.info(f"\n🏗️ Building {architecture} LSTM architecture")

            _ = tf.random.set_seed(42)  # type: ignore[attr-defined]
            np.random.seed(42)

            n_features = len(feature_cols_with_target)
            if architecture == "advanced":
                model = self._build_advanced_lstm(sequence_length, n_features)
            elif architecture == "attention":
                model = self._build_attention_lstm(sequence_length, n_features)
            else:
                model = self._build_standard_lstm(sequence_length, n_features)

            # Enhanced training with callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(  # type: ignore[attr-defined]
                    monitor="val_loss",
                    patience=20,
                    restore_best_weights=True,
                    verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(  # type: ignore[attr-defined]
                    monitor="val_loss", factor=0.2, patience=10, min_lr=1e-7, verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(  # type: ignore[attr-defined]
                    "best_lstm_model.h5",
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                ),
            ]

            # Train model
            logger.info("\n🚀 Training LSTM model")
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1,
                shuffle=False,
            )

            # Evaluate model
            logger.info("\n📊 Evaluating LSTM model")
            train_pred = model.predict(X_train, verbose=0)
            test_pred = model.predict(X_test, verbose=0)

            # Inverse transform predictions
            train_pred_orig = target_scaler.inverse_transform(
                train_pred.reshape(-1, 1)
            ).flatten()
            test_pred_orig = target_scaler.inverse_transform(
                test_pred.reshape(-1, 1)
            ).flatten()
            y_train_orig = target_scaler.inverse_transform(
                y_train.reshape(-1, 1)
            ).flatten()
            y_test_orig = target_scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()

            # Calculate metrics
            metrics = ClimateMetrics.comprehensive_evaluation(
                y_test_orig, test_pred_orig
            )

            # Generate future predictions
            logger.info("\n🔮 Generating future predictions")
            temp_feature_index = 0  # Target is first in feature_cols_with_target
            future_pred = self._generate_lstm_forecast(
                model,
                scaled_features,
                target_scaler,
                sequence_length,
                60,
                temp_feature_index=temp_feature_index,
            )

            # Store results
            self.models["lstm"] = model
            self.scalers["lstm"] = {
                "feature_scaler": feature_scaler,
                "target_scaler": target_scaler,
            }

            self.forecasts["lstm"] = {
                "train_predictions": train_pred_orig,
                "test_predictions": test_pred_orig,
                "y_train": y_train_orig,
                "y_test": y_test_orig,
                "future_predictions": future_pred,
                "training_history": history.history,
                "architecture": architecture,
                "sequence_length": sequence_length,
                "features_used": feature_cols_with_target,
            }

            self.model_performance["lstm"] = {
                **metrics,
                "architecture": architecture,
                "total_parameters": model.count_params(),
                "training_epochs": len(history.history["loss"]),
            }

            logger.info("✅ Enhanced LSTM modeling completed!")
            logger.info(f"📊 Test R²: {metrics['r2']:.4f}")
            logger.info(f"📉 Test RMSE: {metrics['rmse']:.4f}°C")

        except Exception as e:
            logger.error(f"❌ LSTM modeling failed: {e}")
            import traceback

            traceback.print_exc()

    def fit_ensemble_models(self, use_bayesian_optimization: bool = True):
        """Fit ensemble of machine learning models with advanced optimization"""
        logger.info("🎯 Advanced Ensemble Machine Learning Models")
        logger.info("=" * 60)

        try:
            # Prepare data
            X, y = self.prepare_model_data()

            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            logger.info(
                f"Training on {len(X_train)} samples, testing on {len(X_test)} samples"
            )

            # Define base models with enhanced parameter spaces
            models_config = {
                "random_forest": {
                    "model": RandomForestRegressor(random_state=42, n_jobs=-1),
                    "params": (
                        {
                            "n_estimators": (
                                Integer(100, 500) if Integer is not None else [200, 300]
                            ),
                            "max_depth": (
                                Integer(10, 30) if Integer is not None else [15, 20]
                            ),
                            "min_samples_split": (
                                Integer(2, 20) if Integer is not None else [5, 10]
                            ),
                            "min_samples_leaf": (
                                Integer(1, 10) if Integer is not None else [1, 5]
                            ),
                            "max_features": ["sqrt", "log2", None],
                        }
                        if OPTIONAL_DEPS["scikit_optimize"] and Integer is not None
                        else {
                            "n_estimators": [200, 300],
                            "max_depth": [15, 20],
                            "min_samples_split": [5, 10],
                        }
                    ),
                },
                "gradient_boosting": {
                    "model": GradientBoostingRegressor(random_state=42),
                    "params": (
                        {
                            "n_estimators": (
                                Integer(100, 300) if Integer is not None else [150, 200]
                            ),
                            "learning_rate": (
                                Real(0.01, 0.3, prior="log-uniform")
                                if Real is not None
                                else [0.1, 0.15]
                            ),
                            "max_depth": (
                                Integer(3, 10) if Integer is not None else [6, 8]
                            ),
                            "min_samples_split": (
                                Integer(2, 20) if Integer is not None else [2, 10]
                            ),
                            "subsample": (
                                Real(0.8, 1.0) if Real is not None else [0.8, 1.0]
                            ),
                        }
                        if OPTIONAL_DEPS["scikit_optimize"]
                        and Integer is not None
                        and Real is not None
                        else {
                            "n_estimators": [150, 200],
                            "learning_rate": [0.1, 0.15],
                            "max_depth": [6, 8],
                        }
                    ),
                },
            }

            # Add XGBoost if available
            if OPTIONAL_DEPS["xgboost"]:
                models_config["xgboost"] = {
                    "model": xgb.XGBRegressor(random_state=42, n_jobs=-1),  # type: ignore[name-defined]
                    "params": (
                        {
                            "n_estimators": (
                                Integer(100, 300) if Integer is not None else [150, 200]
                            ),
                            "learning_rate": (
                                Real(0.01, 0.3, prior="log-uniform")
                                if Real is not None
                                else [0.1, 0.15]
                            ),
                            "max_depth": (
                                Integer(3, 10) if Integer is not None else [6, 8]
                            ),
                            "min_child_weight": (
                                Integer(1, 10) if Integer is not None else [1, 5]
                            ),
                            "subsample": (
                                Real(0.8, 1.0) if Real is not None else [0.8, 1.0]
                            ),
                            "colsample_bytree": (
                                Real(0.8, 1.0) if Real is not None else [0.8, 1.0]
                            ),
                        }
                        if OPTIONAL_DEPS["scikit_optimize"]
                        else {
                            "n_estimators": [150, 200],
                            "learning_rate": [0.1, 0.15],
                            "max_depth": [6, 8],
                        }
                    ),
                }

            # Train individual models
            trained_models = {}
            model_scores = {}

            for model_name, config in models_config.items():
                logger.info(f"\n🔧 Training {model_name}")

                # Hyperparameter optimization
                if (
                    use_bayesian_optimization
                    and OPTIONAL_DEPS["scikit_optimize"]
                    and BayesSearchCV is not None
                ):
                    search = BayesSearchCV(
                        config["model"],
                        config["params"],
                        cv=tscv,
                        scoring="neg_mean_squared_error",
                        n_iter=30,
                        random_state=42,
                        n_jobs=-1,
                    )
                else:
                    from sklearn.model_selection import GridSearchCV

                    search = GridSearchCV(
                        config["model"],
                        config["params"],
                        cv=tscv,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1,
                    )

                _ = search.fit(X_train, y_train)
                best_model = search.best_estimator_

                # Evaluate model
                train_pred = best_model.predict(X_train)
                test_pred = best_model.predict(X_test)

                # Get feature importance
                if hasattr(best_model, "feature_importances_"):
                    feature_importance = pd.DataFrame(
                        {
                            "feature": X.columns,
                            "importance": best_model.feature_importances_,
                        }
                    ).sort_values("importance", ascending=False)
                else:
                    feature_importance = None

                # Calculate comprehensive metrics
                metrics = ClimateMetrics.comprehensive_evaluation(
                    np.asarray(y_test.values), test_pred
                )

                trained_models[model_name] = best_model
                model_scores[model_name] = metrics["r2"]

                self.models[model_name] = best_model
                self.forecasts[model_name] = {
                    "train_predictions": train_pred,
                    "test_predictions": test_pred,
                    "y_train": y_train.values,
                    "y_test": y_test.values,
                    "best_params": search.best_params_,
                    "feature_importance": feature_importance,
                    "cv_score": search.best_score_,
                }

                self.model_performance[model_name] = {
                    **metrics,
                    "cv_score": search.best_score_,
                    "best_params": search.best_params_,
                }

                logger.info(
                    f"✅ {model_name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}"
                )

            # Create ensemble model
            logger.info("\n🎯 Creating ensemble model")

            # Weight models by their performance
            weights = [max(0.0, model_scores[name]) for name in trained_models.keys()]
            total_weight = sum(weights)
            if total_weight <= 0 or any(np.isnan(weights)):
                normalized_weights = [1.0 / max(1, len(weights))] * len(weights)
            else:
                normalized_weights = [w / total_weight for w in weights]

            ensemble_models = [(name, model) for name, model in trained_models.items()]
            voting_ensemble = VotingRegressor(
                estimators=ensemble_models, weights=normalized_weights
            )

            _ = voting_ensemble.fit(X_train, y_train)

            # Evaluate ensemble
            ensemble_train_pred = voting_ensemble.predict(X_train)
            ensemble_test_pred = voting_ensemble.predict(X_test)

            ensemble_metrics = ClimateMetrics.comprehensive_evaluation(
                np.asarray(y_test.values), ensemble_test_pred
            )

            self.models["ensemble"] = voting_ensemble
            self.forecasts["ensemble"] = {
                "train_predictions": ensemble_train_pred,
                "test_predictions": ensemble_test_pred,
                "y_train": y_train.values,
                "y_test": y_test.values,
                "component_models": list(trained_models.keys()),
                "model_weights": dict(zip(trained_models.keys(), normalized_weights)),
            }

            self.model_performance["ensemble"] = {
                **ensemble_metrics,
                "component_models": list(trained_models.keys()),
                "improvement_over_best": ensemble_metrics["r2"]
                - max(model_scores.values()),
            }

            logger.info("✅ Ensemble modeling completed!")
            logger.info(f"📊 Ensemble R²: {ensemble_metrics['r2']:.4f}")
            logger.info(f"📈 Best individual R²: {max(model_scores.values()):.4f}")

        except Exception as e:
            logger.error(f"❌ Ensemble modeling failed: {e}")
            import traceback

            traceback.print_exc()

    def generate_comprehensive_forecasts(self, years: int = FORECAST_YEARS):
        """Generate comprehensive forecasts from all fitted models"""
        logger.info(f"🔮 Generating comprehensive {years}-year forecasts")
        logger.info("=" * 60)

        consolidated_forecasts = {}

        for model_name in self.models.keys():
            if model_name in ["arima", "sarima"]:
                # Time series forecasts already generated
                if "annual_forecasts" in self.forecasts[model_name]:
                    annual_forecasts = self.forecasts[model_name]["annual_forecasts"]
                    consolidated_forecasts[model_name] = {
                        "annual_temperatures": annual_forecasts[:years],
                        "years": list(
                            range(2025, 2025 + len(annual_forecasts[:years]))
                        ),
                        "method": "time_series",
                        "confidence_intervals": True,
                    }

            elif model_name == "lstm":
                # LSTM forecasts
                if "future_predictions" in self.forecasts[model_name]:
                    future_pred = self.forecasts[model_name]["future_predictions"]
                    # Convert monthly to annual
                    annual_forecasts = []
                    for year in range(years):
                        year_start = year * 12
                        year_end = (year + 1) * 12
                        if year_end <= len(future_pred):
                            annual_avg = future_pred[year_start:year_end].mean()
                            annual_forecasts.append(annual_avg)

                    consolidated_forecasts[model_name] = {
                        "annual_temperatures": annual_forecasts,
                        "years": list(range(2025, 2025 + len(annual_forecasts))),
                        "method": "deep_learning",
                        "confidence_intervals": False,
                    }

            elif model_name in [
                "random_forest",
                "gradient_boosting",
                "xgboost",
                "ensemble",
            ]:
                # ML model forecasts (simple projection)
                if "y_test" in self.forecasts[model_name]:
                    recent_temp = np.mean(self.forecasts[model_name]["y_test"][-365:])

                    # Simple linear projection based on historical trend
                    historical_trend = 0.02  # Approximate warming trend °C/year
                    annual_forecasts = []

                    for year in range(years):
                        projected_temp = recent_temp + historical_trend * (year + 1)
                        annual_forecasts.append(projected_temp)

                    consolidated_forecasts[model_name] = {
                        "annual_temperatures": annual_forecasts,
                        "years": list(range(2025, 2025 + years)),
                        "method": "machine_learning_projection",
                        "confidence_intervals": False,
                    }

        self.consolidated_forecasts = consolidated_forecasts

        # Generate forecast summary
        logger.info("\n📊 Forecast Summary:")
        for model_name, forecast in consolidated_forecasts.items():
            if forecast["annual_temperatures"]:
                avg_temp = np.mean(forecast["annual_temperatures"])
                historical_avg = self.data[TEMP_COLUMN].mean()
                increase = avg_temp - historical_avg
                logger.info(f"   {model_name}: {avg_temp:.2f}°C (+{increase:.2f}°C)")

        return consolidated_forecasts

    def create_comprehensive_visualizations(self, save_path: Optional[str] = None):
        """Create comprehensive visualization dashboard"""
        logger.info("📊 Creating comprehensive model visualization dashboard")

        fig = plt.figure(figsize=(20, 16), dpi=DPI)
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

        # 1. Time series forecasts comparison
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_time_series_forecasts(ax1)

        # 2. Model performance comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_model_performance_comparison(ax2)

        # 3. Feature importance (ensemble)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_feature_importance(ax3)

        # 4. Residual analysis
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_residual_analysis(ax4)

        # 5. Seasonal forecast patterns
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_seasonal_forecasts(ax5)

        # 6. Uncertainty quantification
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_uncertainty_analysis(ax6)

        # 7. Climate risk assessment
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_climate_risk_assessment(ax7)

        # 8. Model ensemble weights
        ax8 = fig.add_subplot(gs[3, 0])
        self._plot_ensemble_weights(ax8)

        # 9. Prediction intervals
        ax9 = fig.add_subplot(gs[3, 1])
        self._plot_prediction_intervals(ax9)

        # 10. Summary statistics
        ax10 = fig.add_subplot(gs[3, 2])
        self._plot_summary_statistics(ax10)

        _ = plt.suptitle(
            "Comprehensive Climate Prediction Analysis Dashboard",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            y=0.98,
        )

        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
            logger.info(f"Dashboard saved to {save_path}")

        plt.show()

        return fig

    def get_model_comparison_report(self) -> str:
        """Generate comprehensive model comparison report"""
        if not self.model_performance:
            return "No models have been trained yet."

        report = "🔬 COMPREHENSIVE MODEL COMPARISON REPORT\n"
        report += "=" * 70 + "\n\n"

        # Performance summary table
        report += "📊 MODEL PERFORMANCE SUMMARY\n"
        report += "-" * 40 + "\n"

        performance_data = []
        for model_name, metrics in self.model_performance.items():
            performance_data.append(
                {
                    "Model": model_name.upper(),
                    "R²": f"{metrics.get('r2', 0):.4f}",
                    "RMSE": f"{metrics.get('rmse', 0):.4f}°C",
                    "MAE": f"{metrics.get('mae', 0):.4f}°C",
                    "MAPE": f"{metrics.get('mape', 0):.2f}%",
                    "Extreme F1": f"{metrics.get('extreme_f1_score', 0):.3f}",
                }
            )

        # Create performance table
        if performance_data:
            df = pd.DataFrame(performance_data)
            report += df.to_string(index=False) + "\n\n"

        # Best performing models
        if self.model_performance:
            best_r2_model = max(
                self.model_performance.items(), key=lambda x: x[1].get("r2", 0)
            )
            best_rmse_model = min(
                self.model_performance.items(),
                key=lambda x: x[1].get("rmse", float("inf")),
            )

            report += "🏆 BEST PERFORMING MODELS\n"
            report += "-" * 30 + "\n"
            report += f"Highest R²: {best_r2_model[0].upper()} ({best_r2_model[1]['r2']:.4f})\n"
            report += f"Lowest RMSE: {best_rmse_model[0].upper()} ({best_rmse_model[1]['rmse']:.4f}°C)\n\n"

        # Forecast comparison
        if hasattr(self, "consolidated_forecasts"):
            report += "🔮 FORECAST COMPARISON (2025-2029)\n"
            report += "-" * 40 + "\n"

            historical_avg = self.data[TEMP_COLUMN].mean()

            for model_name, forecast in self.consolidated_forecasts.items():
                if forecast["annual_temperatures"]:
                    avg_forecast = np.mean(forecast["annual_temperatures"])
                    increase = avg_forecast - historical_avg
                    report += f"{model_name.upper()}: {avg_forecast:.2f}°C (+{increase:.2f}°C)\n"

            report += f"\nHistorical Average (1972-2024): {historical_avg:.2f}°C\n\n"

        # Model-specific insights
        report += "🔍 MODEL-SPECIFIC INSIGHTS\n"
        report += "-" * 35 + "\n"

        for model_name, metrics in self.model_performance.items():
            report += f"\n{model_name.upper()}:\n"

            if model_name in ["arima", "sarima"]:
                report += "  • Type: Time Series Forecasting\n"
                report += "  • Strengths: Captures temporal patterns and seasonality\n"
                report += f"  • AIC: {self.forecasts[model_name]['model_summary'].get('aic', 'N/A')}\n"

            elif model_name == "lstm":
                report += "  • Type: Deep Learning (Neural Network)\n"
                report += "  • Strengths: Complex pattern recognition\n"
                report += (
                    f"  • Architecture: {metrics.get('architecture', 'Standard')}\n"
                )
                report += f"  • Parameters: {metrics.get('total_parameters', 0):,}\n"

            elif model_name == "ensemble":
                report += "  • Type: Ensemble (Multiple Models)\n"
                report += "  • Strengths: Combines multiple approaches\n"
                if "component_models" in metrics:
                    report += (
                        f"  • Components: {', '.join(metrics['component_models'])}\n"
                    )

            else:
                report += "  • Type: Machine Learning\n"
                report += "  • Strengths: Feature-based prediction\n"

            # Common metrics
            report += f"  • Extreme Event Skill (F1): {metrics.get('extreme_f1_score', 0):.3f}\n"
            report += f"  • Directional Accuracy: {metrics.get('directional_accuracy', 0):.3f}\n"

        # Recommendations
        report += "\n💡 RECOMMENDATIONS\n"
        report += "-" * 25 + "\n"

        if self.model_performance:
            best_overall = max(
                self.model_performance.items(), key=lambda x: x[1].get("r2", 0)
            )

            report += (
                f"1. Primary Model: Use {best_overall[0].upper()} for main forecasts\n"
            )

            if "ensemble" in self.model_performance:
                report += (
                    "2. Ensemble Approach: Consider ensemble for robust predictions\n"
                )

            report += (
                "3. Uncertainty: Always report confidence intervals when available\n"
            )
            report += "4. Validation: Regularly update models with new data\n"
            report += (
                "5. Extreme Events: Monitor extreme temperature prediction skill\n"
            )

        report += "\n" + "=" * 70 + "\n"

        return report

    # Helper methods for internal use
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Test time series stationarity"""
        results = {}

        # Augmented Dickey-Fuller test
        adf_result = adfuller(series, autolag="AIC")
        adf_stat = adf_result[0]
        adf_p = adf_result[1]
        adf_lags = adf_result[2]
        # adf_nobs = adf_result[3]  # Not used
        adf_critical = adf_result[4] if len(adf_result) > 4 else {}
        results["adf"] = {
            "statistic": adf_stat,
            "p_value": adf_p,
            "lags_used": adf_lags,
            "critical_values": adf_critical,
            "is_stationary": adf_p <= 0.05,
        }

        # KPSS test
        kpss_stat, kpss_p, kpss_lags, kpss_critical = kpss(series, regression="ct")
        results["kpss"] = {
            "statistic": kpss_stat,
            "p_value": kpss_p,
            "lags_used": kpss_lags,
            "critical_values": kpss_critical,
            "is_stationary": kpss_p > 0.05,
        }

        return results

    def _model_diagnostics(self, fitted_model, original_series) -> Dict[str, Any]:
        """Perform comprehensive model diagnostics"""
        diagnostics = {}

        try:
            residuals = fitted_model.resid

            # Ljung-Box test for serial correlation
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            diagnostics["ljung_box_pvalue"] = ljung_box["lb_pvalue"].iloc[-1]

            # Jarque-Bera test for normality
            _, jb_pvalue = stats.jarque_bera(residuals.dropna())
            diagnostics["jarque_bera_pvalue"] = jb_pvalue

            # Residual statistics
            diagnostics["residual_mean"] = residuals.mean()
            diagnostics["residual_std"] = residuals.std()
            diagnostics["residual_skewness"] = residuals.skew()
            diagnostics["residual_kurtosis"] = residuals.kurtosis()

        except Exception as e:
            logger.warning(f"Some diagnostics failed: {e}")

        return diagnostics

    def _create_sequences(
        self, features: np.ndarray, target: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i - sequence_length : i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def _build_advanced_lstm(self, sequence_length: int, n_features: int) -> Any:
        """Build advanced LSTM with multiple layers and regularization"""
        model = tf.keras.Sequential(  # type: ignore[attr-defined]
            [
                tf.keras.layers.LSTM(  # type: ignore[attr-defined]
                    128,
                    return_sequences=True,
                    input_shape=(sequence_length, n_features),
                    dropout=0.2,
                    recurrent_dropout=0.2,
                ),
                tf.keras.layers.BatchNormalization(),  # type: ignore[attr-defined]
                tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),  # type: ignore[attr-defined]
                tf.keras.layers.BatchNormalization(),  # type: ignore[attr-defined]
                tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),  # type: ignore[attr-defined]
                tf.keras.layers.BatchNormalization(),  # type: ignore[attr-defined]
                tf.keras.layers.Dense(  # type: ignore[attr-defined]
                    64, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)  # type: ignore[attr-defined]
                ),
                tf.keras.layers.Dropout(0.3),  # type: ignore[attr-defined]
                tf.keras.layers.Dense(32, activation="relu"),  # type: ignore[attr-defined]
                tf.keras.layers.Dropout(0.2),  # type: ignore[attr-defined]
                tf.keras.layers.Dense(1),  # type: ignore[attr-defined]
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),  # type: ignore[attr-defined]
            loss="mse",
            metrics=["mae"],
        )

        return model

    def _build_attention_lstm(self, sequence_length: int, n_features: int) -> Any:
        """Build LSTM with attention mechanism"""
        # Note: This is a simplified attention mechanism
        # For production, consider using Transformer models
        inputs = tf.keras.Input(shape=(sequence_length, n_features))  # type: ignore[attr-defined]

        # LSTM layers
        lstm_out = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)  # type: ignore[attr-defined]
        lstm_out = tf.keras.layers.BatchNormalization()(lstm_out)  # type: ignore[attr-defined]

        # Attention mechanism (simplified)
        attention = tf.keras.layers.Dense(1, activation="tanh")(lstm_out)  # type: ignore[attr-defined]
        attention = tf.keras.layers.Flatten()(attention)  # type: ignore[attr-defined]
        attention = tf.keras.layers.Activation("softmax")(attention)  # type: ignore[attr-defined]
        attention = tf.keras.layers.RepeatVector(64)(attention)  # type: ignore[attr-defined]
        attention = tf.keras.layers.Permute([2, 1])(attention)  # type: ignore[attr-defined]

        # Apply attention
        attended = tf.keras.layers.Multiply()([lstm_out, attention])  # type: ignore[attr-defined]
        attended = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(  # type: ignore[attr-defined]
            attended
        )

        # Output layers
        output = tf.keras.layers.Dense(32, activation="relu")(attended)  # type: ignore[attr-defined]
        output = tf.keras.layers.Dropout(0.2)(output)  # type: ignore[attr-defined]
        output = tf.keras.layers.Dense(1)(output)  # type: ignore[attr-defined]

        model = tf.keras.Model(inputs=inputs, outputs=output)  # type: ignore[attr-defined]

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])  # type: ignore[attr-defined]

        return model

    def _build_standard_lstm(self, sequence_length: int, n_features: int) -> Any:
        """Build standard LSTM model"""
        model = tf.keras.Sequential(  # type: ignore[attr-defined]
            [
                tf.keras.layers.LSTM(  # type: ignore[attr-defined]
                    50, return_sequences=True, input_shape=(sequence_length, n_features)
                ),
                tf.keras.layers.LSTM(25, return_sequences=False),  # type: ignore[attr-defined]
                tf.keras.layers.Dense(25, activation="relu"),  # type: ignore[attr-defined]
                tf.keras.layers.Dense(1),  # type: ignore[attr-defined]
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # type: ignore[attr-defined]
        return model

    def _generate_lstm_forecast(
        self,
        model,
        scaled_features: np.ndarray,
        target_scaler,
        sequence_length: int,
        forecast_steps: int,
        *,
        temp_feature_index: int = 0,
    ) -> np.ndarray:
        """Generate future forecasts using LSTM model"""
        # Use the last sequence as starting point
        last_sequence = scaled_features[-sequence_length:].reshape(
            1, sequence_length, -1
        )

        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(forecast_steps):
            # Predict next value
            pred = model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])

            # Update sequence (simplified - using last features with new prediction)
            next_features = current_sequence[0, -1, :].copy()
            # Update the designated temperature feature with the new prediction
            if 0 <= temp_feature_index < next_features.shape[0]:
                next_features[temp_feature_index] = pred[0, 0]

            # Shift sequence and add new prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = next_features

        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        return target_scaler.inverse_transform(predictions).flatten()

    # Visualization helper methods
    def _plot_time_series_forecasts(self, ax):
        """Plot time series forecasts comparison"""
        if "arima" in self.forecasts:
            arima_data = self.forecasts["arima"]
            historical = arima_data["historical_data"]

            # Plot historical data (last 5 years)
            recent_historical = historical[-60:]
            ax.plot(
                recent_historical.index,
                recent_historical.values,
                "b-",
                linewidth=2,
                label="Historical",
            )

            # Plot ARIMA forecast
            forecast_dates = arima_data["future_dates"]
            forecast_values = arima_data["forecast"]

            ax.plot(
                forecast_dates,
                forecast_values,
                "r-",
                linewidth=2,
                label="ARIMA Forecast",
            )

            # Plot confidence intervals
            ci = arima_data["forecast_ci"]
            ax.fill_between(
                forecast_dates,
                ci.iloc[:, 0],
                ci.iloc[:, 1],
                alpha=0.3,
                color="red",
                label="95% Confidence",
            )

        ax.set_title("Climate Forecasting Results", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_model_performance_comparison(self, ax):
        """Plot model performance comparison"""
        if not self.model_performance:
            ax.text(
                0.5,
                0.5,
                "No performance data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        models = list(self.model_performance.keys())
        r2_scores = [self.model_performance[m].get("r2", 0) for m in models]
        rmse_scores = [self.model_performance[m].get("rmse", 0) for m in models]

        x = np.arange(len(models))

        # Create twin axes
        ax2 = ax.twinx()

        bars1 = ax.bar(x - 0.2, r2_scores, 0.4, label="R²", alpha=0.8, color="blue")
        bars2 = ax2.bar(
            x + 0.2, rmse_scores, 0.4, label="RMSE (°C)", alpha=0.8, color="red"
        )

        ax.set_xlabel("Models")
        ax.set_ylabel("R² Score", color="blue")
        ax2.set_ylabel("RMSE (°C)", color="red")
        ax.set_title("Model Performance Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models], rotation=45)

        # Add value labels
        for bar, score in zip(bars1, r2_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        for bar, score in zip(bars2, rmse_scores):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.grid(True, alpha=0.3)

    def _plot_feature_importance(self, ax):
        """Plot feature importance from ensemble model"""
        if "ensemble" in self.forecasts and "random_forest" in self.forecasts:
            rf_importance = self.forecasts["random_forest"].get("feature_importance")
            if rf_importance is not None and len(rf_importance) > 0:
                top_features = rf_importance.head(10)

                _ = ax.barh(
                    range(len(top_features)),
                    top_features["importance"],
                    color="green",
                    alpha=0.7,
                )
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features["feature"], fontsize=9)
                ax.set_xlabel("Importance")
                ax.set_title("Top 10 Feature Importance", fontweight="bold")

                # Add value labels
                for i, val in enumerate(top_features["importance"]):
                    ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8)

                ax.grid(True, alpha=0.3, axis="x")
                return

        ax.text(
            0.5,
            0.5,
            "Feature importance\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    def _plot_residual_analysis(self, ax):
        """Plot residual analysis for best model"""
        best_model = None
        best_score = -np.inf

        for model_name, performance in self.model_performance.items():
            if performance.get("r2", 0) > best_score:
                best_score = performance["r2"]
                best_model = model_name

        if best_model and best_model in self.forecasts:
            forecast_data = self.forecasts[best_model]
            if "y_test" in forecast_data and "test_predictions" in forecast_data:
                y_true = forecast_data["y_test"]
                y_pred = forecast_data["test_predictions"]

                residuals = y_true - y_pred

                ax.scatter(y_pred, residuals, alpha=0.6, color="purple")
                ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
                ax.set_xlabel("Predicted Temperature (°C)")
                ax.set_ylabel("Residuals (°C)")
                ax.set_title(
                    f"Residual Analysis ({best_model.upper()})", fontweight="bold"
                )
                ax.grid(True, alpha=0.3)

                # Add statistics
                rmse = np.sqrt(np.mean(residuals**2))
                mae = np.mean(np.abs(residuals))
                ax.text(
                    0.05,
                    0.95,
                    f"RMSE: {rmse:.3f}°C\nMAE: {mae:.3f}°C",
                    transform=ax.transAxes,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                return

        ax.text(
            0.5,
            0.5,
            "Residual analysis\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    def _plot_seasonal_forecasts(self, ax):
        """Plot seasonal forecast patterns"""
        if (
            hasattr(self, "consolidated_forecasts")
            and "arima" in self.consolidated_forecasts
        ):
            # This is a placeholder - would need monthly forecast data
            months = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]

            # Use historical seasonal pattern as example
            historical_monthly = self.data.groupby(self.data["timestamp"].dt.month)[
                TEMP_COLUMN
            ].mean()

            ax.plot(
                months,
                historical_monthly.values,
                "b-o",
                linewidth=2,
                markersize=6,
                label="Historical Pattern",
            )

            ax.set_title("Seasonal Temperature Patterns", fontweight="bold")
            ax.set_xlabel("Month")
            ax.set_ylabel("Temperature (°C)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            _ = plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(
                0.5,
                0.5,
                "Seasonal forecasts\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _plot_uncertainty_analysis(self, ax):
        """Plot uncertainty analysis"""
        if "arima" in self.forecasts:
            forecast_data = self.forecasts["arima"]
            if "forecast_ci" in forecast_data:
                ci = forecast_data["forecast_ci"]

                # Calculate uncertainty metrics
                uncertainty = (ci.iloc[:, 1] - ci.iloc[:, 0]) / 2

                ax.plot(uncertainty.values, "g-", linewidth=2)
                ax.set_title("Forecast Uncertainty Over Time", fontweight="bold")
                ax.set_xlabel("Forecast Month")
                ax.set_ylabel("Uncertainty (±°C)")
                ax.grid(True, alpha=0.3)

                # Add statistics
                mean_uncertainty = uncertainty.mean()
                max_uncertainty = uncertainty.max()
                ax.text(
                    0.05,
                    0.95,
                    f"Mean: ±{mean_uncertainty:.2f}°C\nMax: ±{max_uncertainty:.2f}°C",
                    transform=ax.transAxes,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                return

        ax.text(
            0.5,
            0.5,
            "Uncertainty analysis\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    def _plot_climate_risk_assessment(self, ax):
        """Plot climate risk assessment"""
        if hasattr(self, "consolidated_forecasts"):
            # Calculate risk levels based on temperature increases
            risk_data = []
            historical_avg = self.data[TEMP_COLUMN].mean()

            for model_name, forecast in self.consolidated_forecasts.items():
                if forecast["annual_temperatures"]:
                    avg_increase = (
                        np.mean(forecast["annual_temperatures"]) - historical_avg
                    )
                    risk_data.append((model_name, avg_increase))

            if risk_data:
                models, increases = zip(*risk_data)
                colors = [
                    "green" if inc < 0.5 else "orange" if inc < 1.0 else "red"
                    for inc in increases
                ]

                bars = ax.bar(range(len(models)), increases, color=colors, alpha=0.7)
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels([m.upper() for m in models], rotation=45)
                ax.set_ylabel("Temperature Increase (°C)")
                ax.set_title("Climate Risk Assessment (2025-2029)", fontweight="bold")
                ax.grid(True, alpha=0.3)

                # Add risk level lines
                ax.axhline(
                    y=0.5,
                    color="orange",
                    linestyle=":",
                    alpha=0.7,
                    label="Moderate Risk",
                )
                ax.axhline(
                    y=1.0, color="red", linestyle=":", alpha=0.7, label="High Risk"
                )

                # Add value labels
                for bar, inc in zip(bars, increases):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"+{inc:.2f}°C",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

                ax.legend()
                return

        ax.text(
            0.5,
            0.5,
            "Climate risk assessment\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    def _plot_ensemble_weights(self, ax):
        """Plot ensemble model weights"""
        if (
            "ensemble" in self.forecasts
            and "model_weights" in self.forecasts["ensemble"]
        ):
            weights = self.forecasts["ensemble"]["model_weights"]

            models = list(weights.keys())
            weight_values = list(weights.values())

            cmap = plt.get_cmap("Set3")
            colors = cmap(np.linspace(0, 1, len(models)))
            _, _, autotexts = ax.pie(
                weight_values,
                labels=models,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )

            ax.set_title("Ensemble Model Weights", fontweight="bold")

            # Enhance text visibility
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
        else:
            ax.text(
                0.5,
                0.5,
                "Ensemble weights\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _plot_prediction_intervals(self, ax):
        """Plot prediction intervals for multiple models"""
        if hasattr(self, "consolidated_forecasts"):
            years = list(range(2025, 2030))

            for model_name, forecast in self.consolidated_forecasts.items():
                if forecast["annual_temperatures"]:
                    temps = forecast["annual_temperatures"][:5]  # 2025-2029
                    ax.plot(
                        years[: len(temps)],
                        temps,
                        "o-",
                        linewidth=2,
                        markersize=6,
                        label=model_name.upper(),
                        alpha=0.8,
                    )

            # Add historical average line
            historical_avg = self.data[TEMP_COLUMN].mean()
            ax.axhline(
                y=historical_avg,
                color="black",
                linestyle="--",
                alpha=0.7,
                label=f"Historical Avg ({historical_avg:.1f}°C)",
            )

            ax.set_title("Multi-Model Temperature Predictions", fontweight="bold")
            ax.set_xlabel("Year")
            ax.set_ylabel("Temperature (°C)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "Prediction intervals\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _plot_summary_statistics(self, ax):
        """Plot summary statistics"""
        ax.axis("off")

        # Create summary text
        summary_text = "🔬 MODEL SUMMARY STATISTICS\n"
        summary_text += "=" * 35 + "\n\n"

        if self.model_performance:
            # Best performing model
            best_model = max(
                self.model_performance.items(), key=lambda x: x[1].get("r2", 0)
            )

            summary_text += f"🏆 Best Model: {best_model[0].upper()}\n"
            summary_text += f"   R²: {best_model[1].get('r2', 0):.4f}\n"
            summary_text += f"   RMSE: {best_model[1].get('rmse', 0):.3f}°C\n\n"

            # Model count
            summary_text += f"📊 Models Trained: {len(self.model_performance)}\n"

            # Average performance
            avg_r2 = np.mean([m.get("r2", 0) for m in self.model_performance.values()])
            avg_rmse = np.mean(
                [m.get("rmse", 0) for m in self.model_performance.values()]
            )

            summary_text += f"   Average R²: {avg_r2:.4f}\n"
            summary_text += f"   Average RMSE: {avg_rmse:.3f}°C\n\n"

            # Forecast summary
            if hasattr(self, "consolidated_forecasts"):
                historical_avg = self.data[TEMP_COLUMN].mean()
                forecast_increases = []

                for forecast in self.consolidated_forecasts.values():
                    if forecast["annual_temperatures"]:
                        avg_temp = np.mean(forecast["annual_temperatures"])
                        increase = avg_temp - historical_avg
                        forecast_increases.append(increase)

                if forecast_increases:
                    consensus_increase = np.mean(forecast_increases)
                    summary_text += "🌡️ Consensus Forecast:\n"
                    summary_text += (
                        f"   Temperature Increase: +{consensus_increase:.2f}°C\n"
                    )
                    summary_text += f"   Range: {min(forecast_increases):.2f} to {max(forecast_increases):.2f}°C\n\n"

            # Data quality
            total_samples = len(self.data)
            date_range = (
                self.data["timestamp"].max() - self.data["timestamp"].min()
            ).days

            summary_text += "📅 Dataset Information:\n"
            summary_text += f"   Total Samples: {total_samples:,}\n"
            summary_text += f"   Time Span: {date_range:,} days\n"
            summary_text += f"   Years: {date_range / 365.25:.1f}\n"

        else:
            summary_text += "No model performance data available."

        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )


# Convenience function for quick modeling
def quick_climate_forecast(
    data: pd.DataFrame,
    tree_loss_by_year: pd.DataFrame,
    models: List[str] = ["arima", "lstm", "ensemble"],
) -> EnhancedClimatePredictor:
    """
    Quick setup for climate forecasting with multiple models

    Args:
        data: Daily climate data
        tree_loss_by_year: Annual deforestation data
        models: List of models to train ['arima', 'sarima', 'lstm', 'ensemble']

    Returns:
        Trained EnhancedClimatePredictor instance
    """
    logger.info("🚀 Quick Climate Forecasting Setup")

    predictor = EnhancedClimatePredictor(data, tree_loss_by_year)
    _ = predictor.prepare_enhanced_features()

    for model in models:
        if model == "arima":
            predictor.fit_enhanced_arima_sarima(seasonal=False)
        elif model == "sarima":
            predictor.fit_enhanced_arima_sarima(seasonal=True)
        elif model == "lstm":
            predictor.fit_enhanced_lstm()
        elif model == "ensemble":
            predictor.fit_ensemble_models()
        else:
            logger.warning(f"Unknown model type: {model}")

    # Generate forecasts
    _ = predictor.generate_comprehensive_forecasts()

    return predictor
