"""
Improved ARIMA/SARIMA Module
===========================

Enhanced time series forecasting with proper stationarity testing,
differencing, and diagnostic checks for climate data analysis.

"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

from data_dictionary import TEMPERATURE_COLUMNS

warnings.filterwarnings("ignore")

try:
    from pmdarima import auto_arima

    PMDARIMA_AVAILABLE = True
except ImportError:
    auto_arima = None  # type: ignore
    PMDARIMA_AVAILABLE = False
    print("WARNING: pmdarima not available. Using manual parameter selection.")


class ImprovedTimeSeriesPredictor:
    """
    Enhanced time series forecasting with proper statistical validation
    """

    def __init__(self, data, tree_loss_by_year=None):
        """
        Initialize predictor with validated data

        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with temperature measurements
        tree_loss_by_year : pd.DataFrame, optional
            Deforestation data (not used for improved ARIMA)
        """
        self.data = data
        self.models = {}
        self.forecasts = {}
        self.diagnostics = {}

        # Use proper temperature column
        self.temp_col = TEMPERATURE_COLUMNS["daily_mean"]
        if self.temp_col not in data.columns:
            raise ValueError(f"Temperature column {self.temp_col} not found in data")

        print(f"Initialized with {len(data):,} records using {self.temp_col}")

    def prepare_monthly_series(self):
        """
        Prepare monthly temperature series for ARIMA modeling

        Returns:
        --------
        pd.Series : Monthly temperature series with datetime index
        """
        print("Preparing monthly temperature series...")

        # Create monthly aggregation
        data_copy = self.data.copy()
        data_copy["YearMonth"] = data_copy["timestamp"].dt.to_period("M")

        # Calculate monthly averages
        monthly_temp = data_copy.groupby("YearMonth")[self.temp_col].agg(
            ["mean", "count"]
        )

        # Filter months with sufficient data (at least 20 days)
        monthly_temp = monthly_temp[monthly_temp["count"] >= 20]

        # Convert to series with datetime index
        monthly_series = monthly_temp["mean"]
        monthly_series.index = monthly_series.index.to_timestamp()

        print(f"Monthly series prepared: {len(monthly_series)} months")
        print(
            f"Date range: {monthly_series.index.min()} to {monthly_series.index.max()}"
        )
        print(f"Mean temperature: {monthly_series.mean():.2f}°C")
        print(
            f"Temperature range: {monthly_series.min():.2f}°C to {monthly_series.max():.2f}°C"
        )

        self.monthly_series = monthly_series
        return monthly_series

    def test_stationarity(self, series, alpha=0.05):
        """
        Test for stationarity using ADF and KPSS tests

        Parameters:
        -----------
        series : pd.Series
            Time series to test
        alpha : float
            Significance level

        Returns:
        --------
        dict : Stationarity test results
        """
        print("Testing for stationarity...")

        results = {
            "adf_test": {},
            "kpss_test": {},
            "is_stationary": False,
            "recommendation": "",
        }

        # Augmented Dickey-Fuller test (H0: non-stationary)
        adf_result = adfuller(series, autolag="AIC")
        # Properly unpack adfuller results
        adf_statistic, adf_pvalue, adf_lags, adf_nobs = adf_result[:4]
        adf_critical_values = adf_result[4] if len(adf_result) > 4 else {}

        results["adf_test"] = {
            "statistic": adf_statistic,
            "p_value": adf_pvalue,
            "critical_values": adf_critical_values,
            "is_stationary": adf_pvalue <= alpha,
        }

        # KPSS test (H0: stationary)
        kpss_result = kpss(series, regression="ct")
        results["kpss_test"] = {
            "statistic": kpss_result[0],
            "p_value": kpss_result[1],
            "critical_values": kpss_result[3],
            "is_stationary": kpss_result[1] > alpha,
        }

        # Determine overall stationarity
        adf_stationary = results["adf_test"]["is_stationary"]
        kpss_stationary = results["kpss_test"]["is_stationary"]

        if adf_stationary and kpss_stationary:
            results["is_stationary"] = True
            results["recommendation"] = "Series appears stationary"
        elif not adf_stationary and not kpss_stationary:
            results["is_stationary"] = False
            results["recommendation"] = "Series is non-stationary, needs differencing"
        else:
            results["is_stationary"] = False
            results["recommendation"] = "Mixed signals, proceed with caution"

        print(
            f"ADF Test: p-value = {results['adf_test']['p_value']:.4f} (stationary: {adf_stationary})"
        )
        print(
            f"KPSS Test: p-value = {results['kpss_test']['p_value']:.4f} (stationary: {kpss_stationary})"
        )
        print(f"Recommendation: {results['recommendation']}")

        return results

    def find_differencing_order(self, series, max_d=3):
        """
        Find appropriate differencing order to achieve stationarity

        Parameters:
        -----------
        series : pd.Series
            Time series to analyze
        max_d : int
            Maximum differencing order to test

        Returns:
        --------
        int : Recommended differencing order
        """
        print("Finding optimal differencing order...")

        current_series = series.copy()

        for d in range(max_d + 1):
            stationarity_results = self.test_stationarity(current_series)

            print(f"Differencing order {d}: {stationarity_results['recommendation']}")

            if stationarity_results["is_stationary"] or d == max_d:
                return d

            # Apply another round of differencing
            current_series = current_series.diff().dropna()

        return max_d

    def seasonal_decomposition(self, series, period=12):
        """
        Perform seasonal decomposition analysis

        Parameters:
        -----------
        series : pd.Series
            Time series to decompose
        period : int
            Seasonal period (12 for monthly data)

        Returns:
        --------
        statsmodels.tsa.seasonal.DecomposeResult : Decomposition results
        """
        print(f"Performing seasonal decomposition (period={period})...")

        # Ensure we have enough data for seasonal decomposition
        if len(series) < 2 * period:
            print(
                f"WARNING: Series too short for seasonal decomposition (need at least {2*period} observations)"
            )
            return None

        try:
            decomposition = seasonal_decompose(series, model="additive", period=period)

            # Calculate seasonal strength
            seasonal_strength = np.var(decomposition.seasonal) / (
                np.var(decomposition.seasonal) + np.var(decomposition.resid)
            )
            trend_strength = np.var(decomposition.trend.dropna()) / (
                np.var(decomposition.trend.dropna()) + np.var(decomposition.resid)
            )

            print(f"Seasonal strength: {seasonal_strength:.3f}")
            print(f"Trend strength: {trend_strength:.3f}")

            # Store diagnostics
            self.diagnostics["decomposition"] = {
                "seasonal_strength": seasonal_strength,
                "trend_strength": trend_strength,
                "residual_std": np.std(decomposition.resid.dropna()),
            }

            return decomposition

        except Exception as e:
            print(f"Seasonal decomposition failed: {e}")
            return None

    def fit_improved_arima(self, series=None, max_p=5, max_d=2, max_q=5):
        """
        Fit ARIMA model with proper stationarity testing and diagnostics

        Parameters:
        -----------
        series : pd.Series, optional
            Time series (uses monthly series if not provided)
        max_p, max_d, max_q : int
            Maximum ARIMA parameters to consider

        Returns:
        --------
        dict : Model results and diagnostics
        """
        print("=" * 70)
        print("IMPROVED ARIMA MODEL FITTING")
        print("=" * 70)

        if series is None:
            if not hasattr(self, "monthly_series"):
                self.prepare_monthly_series()
            series = self.monthly_series

        # 1. Test stationarity
        stationarity_results = self.test_stationarity(series)

        # 2. Find appropriate differencing order
        if not stationarity_results["is_stationary"]:
            d_order = self.find_differencing_order(series, max_d)
            print(f"Recommended differencing order: {d_order}")
        else:
            d_order = 0
            print("Series is stationary, no differencing needed")

        # 3. Seasonal decomposition
        decomposition = self.seasonal_decomposition(series)

        # 4. Model selection
        if PMDARIMA_AVAILABLE and auto_arima is not None:
            print("Using auto_arima for parameter selection...")
            try:
                auto_model = auto_arima(
                    series,
                    start_p=0,
                    start_q=0,
                    max_p=max_p,
                    max_q=max_q,
                    d=None,
                    max_d=max_d,  # Let auto_arima determine d
                    seasonal=False,  # Start with non-seasonal
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=True,
                )

                order = auto_model.order
                print(f"Auto-selected ARIMA order: {order}")

            except Exception as e:
                print(f"auto_arima failed: {e}")
                order = (1, d_order, 1)  # Fallback

        else:
            # Manual parameter selection
            order = (2, d_order, 2)  # Conservative choice
            print(f"Using manual ARIMA order: {order}")

        # 5. Fit ARIMA model
        try:
            arima_model = ARIMA(series, order=order)
            arima_fitted = arima_model.fit()  # type: ignore

            print(f"ARIMA{order} fitted successfully")
            print(f"AIC: {arima_fitted.aic:.2f}")
            print(f"BIC: {arima_fitted.bic:.2f}")
            print(f"Log-likelihood: {arima_fitted.llf:.2f}")

            # 6. Model diagnostics
            diagnostics = self.run_diagnostics(arima_fitted, series)

            # 7. Generate forecasts
            forecast_steps = 72  # 6 years of monthly data
            forecast_result = arima_fitted.get_forecast(steps=forecast_steps)
            forecast = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()

            # Create future dates
            last_date = series.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_steps,
                freq="M",
            )

            # Calculate annual forecasts
            annual_forecasts = {}
            for year_offset in range(6):
                year_start = year_offset * 12
                year_end = (year_offset + 1) * 12
                annual_avg = forecast[year_start:year_end].mean()
                year_label = str(2025 + year_offset)
                annual_forecasts[year_label] = float(annual_avg)

            print("\\nAnnual Temperature Forecasts (ARIMA):")
            historical_avg = series.mean()
            for year, temp in annual_forecasts.items():
                increase = temp - historical_avg
                print(f"  {year}: {temp:.2f}°C ({increase:+.2f}°C from historical)")

            # Store results
            arima_results = {
                "model": arima_fitted,
                "order": order,
                "forecast": forecast,
                "forecast_dates": future_dates,
                "confidence_interval": forecast_ci,
                "annual_forecasts": annual_forecasts,
                "historical_series": series,
                "stationarity_tests": stationarity_results,
                "diagnostics": diagnostics,
                "decomposition": decomposition,
                "model_summary": {
                    "aic": arima_fitted.aic,
                    "bic": arima_fitted.bic,
                    "llf": arima_fitted.llf,
                    "historical_mean": historical_avg,
                    "forecast_mean": forecast.mean(),
                    "forecast_increase": forecast.mean() - historical_avg,
                },
            }

            self.models["improved_arima"] = arima_results

            print("ARIMA modeling completed successfully!")
            return arima_results

        except Exception as e:
            print(f"ARIMA model fitting failed: {e}")
            return None

    def fit_improved_sarima(self, series=None, seasonal_period=12):
        """
        Fit SARIMA model with seasonal components

        Parameters:
        -----------
        series : pd.Series, optional
            Time series (uses monthly series if not provided)
        seasonal_period : int
            Seasonal period (12 for monthly data)

        Returns:
        --------
        dict : Model results and diagnostics
        """
        print("=" * 70)
        print("IMPROVED SARIMA MODEL FITTING")
        print("=" * 70)

        if series is None:
            if not hasattr(self, "monthly_series"):
                self.prepare_monthly_series()
            series = self.monthly_series

        # Check if we have enough data for seasonal modeling
        if len(series) < 3 * seasonal_period:
            print(
                f"WARNING: Insufficient data for SARIMA (need at least {3*seasonal_period} observations)"
            )
            return self.fit_improved_arima(series)  # Fall back to ARIMA

        # Seasonal stationarity testing
        print(f"Testing seasonal stationarity (period={seasonal_period})...")

        # Test seasonal differencing if needed
        seasonal_diff_series = series.diff(seasonal_period).dropna()
        self.test_stationarity(seasonal_diff_series)

        # Model selection with auto_arima if available
        if PMDARIMA_AVAILABLE and auto_arima is not None:
            try:
                print("Using auto_arima for SARIMA parameter selection...")
                auto_sarima = auto_arima(
                    series,
                    start_p=0,
                    start_q=0,
                    start_P=0,
                    start_Q=0,
                    max_p=3,
                    max_q=3,
                    max_P=2,
                    max_Q=2,
                    seasonal=True,
                    m=seasonal_period,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=True,
                )

                order = auto_sarima.order
                seasonal_order = auto_sarima.seasonal_order

                print(
                    f"Auto-selected SARIMA order: {order} x {seasonal_order}{seasonal_period}"
                )

            except Exception as e:
                print(f"auto_arima SARIMA failed: {e}")
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, seasonal_period)
        else:
            # Manual seasonal parameters
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, seasonal_period)
            print(f"Using manual SARIMA order: {order} x {seasonal_order}")

        # Fit SARIMA model
        try:
            sarima_model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            sarima_fitted = sarima_model.fit(disp=False)  # type: ignore

            print("SARIMA fitted successfully")
            if hasattr(sarima_fitted, "aic") and hasattr(sarima_fitted, "bic"):
                print(f"AIC: {sarima_fitted.aic:.2f}")  # type: ignore
                print(f"BIC: {sarima_fitted.bic:.2f}")  # type: ignore
            else:
                print(
                    "SARIMA fit did not return a model with AIC/BIC attributes. The result may be an array or tuple, not a fitted model object."
                )

            # Model diagnostics
            diagnostics = self.run_diagnostics(sarima_fitted, series)

            # Generate forecasts
            forecast_steps = 72
            # Ensure sarima_fitted is a SARIMAXResults object before calling get_forecast
            from statsmodels.tsa.statespace.sarimax import SARIMAXResults

            if isinstance(sarima_fitted, SARIMAXResults) or hasattr(
                sarima_fitted, "get_forecast"
            ):
                forecast_result = sarima_fitted.get_forecast(steps=forecast_steps)  # type: ignore
                forecast = forecast_result.predicted_mean
                forecast_ci = forecast_result.conf_int()
            else:
                print(
                    "SARIMA fit did not return a valid results object; using np.nan for forecasts."
                )
                forecast = pd.Series([np.nan] * forecast_steps)
                forecast_ci = pd.DataFrame(
                    {
                        "lower": [np.nan] * forecast_steps,
                        "upper": [np.nan] * forecast_steps,
                    }
                )

            # Future dates and annual forecasts
            last_date = series.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_steps,
                freq="M",
            )

            annual_forecasts = {}
            for year_offset in range(6):
                year_start = year_offset * 12
                year_end = (year_offset + 1) * 12
                annual_avg = forecast[year_start:year_end].mean()
                year_label = str(2025 + year_offset)
                annual_forecasts[year_label] = float(annual_avg)

            print("\\nAnnual Temperature Forecasts (SARIMA):")
            historical_avg = series.mean()
            for year, temp in annual_forecasts.items():
                increase = temp - historical_avg
                print(f"  {year}: {temp:.2f}°C ({increase:+.2f}°C from historical)")

            # Store results
            sarima_results = {
                "model": sarima_fitted,
                "order": order,
                "seasonal_order": seasonal_order,
                "forecast": forecast,
                "forecast_dates": future_dates,
                "confidence_interval": forecast_ci,
                "annual_forecasts": annual_forecasts,
                "historical_series": series,
                "diagnostics": diagnostics,
                "model_summary": {
                    "aic": sarima_fitted.aic,  # type: ignore
                    "bic": sarima_fitted.bic,  # type: ignore
                    "llf": sarima_fitted.llf,  # type: ignore
                    "historical_mean": historical_avg,
                    "forecast_mean": forecast.mean(),
                    "forecast_increase": forecast.mean() - historical_avg,
                },
            }

            self.models["improved_sarima"] = sarima_results

            print("SARIMA modeling completed successfully!")
            return sarima_results

        except Exception as e:
            print(f"SARIMA model fitting failed: {e}")
            return None

    def run_diagnostics(self, fitted_model, original_series):
        """
        Run comprehensive model diagnostics

        Parameters:
        -----------
        fitted_model : statsmodels fitted model
            The fitted ARIMA/SARIMA model
        original_series : pd.Series
            Original time series

        Returns:
        --------
        dict : Diagnostic test results
        """
        print("Running model diagnostics...")

        diagnostics = {}

        try:
            # Residual tests
            residuals = fitted_model.resid

            # Ljung-Box test for autocorrelation in residuals
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            diagnostics["ljung_box"] = {
                "statistics": lb_test["lb_stat"].tolist(),
                "p_values": lb_test["lb_pvalue"].tolist(),
                "passes": (lb_test["lb_pvalue"] > 0.05).all(),
            }

            # Normality test of residuals
            jb_test = stats.jarque_bera(residuals)

            # Extract values from jarque_bera test result
            # Handle both named tuple (newer scipy) and regular tuple (older scipy)
            jb_statistic_raw = getattr(jb_test, "statistic", jb_test[0])
            jb_pvalue_raw = getattr(jb_test, "pvalue", jb_test[1])

            # Ensure conversion to float for type safety
            jb_statistic_float = float(jb_statistic_raw)  # type: ignore
            jb_pvalue_float = float(jb_pvalue_raw)  # type: ignore

            diagnostics["jarque_bera"] = {
                "statistic": jb_statistic_float,
                "p_value": jb_pvalue_float,
                "passes": jb_pvalue_float > 0.05,
            }

            # Heteroscedasticity test (variables kept for potential future use)
            _residuals_squared = residuals**2
            _residuals_abs = np.abs(residuals)

            diagnostics["residual_stats"] = {
                "mean": np.mean(residuals),
                "std": np.std(residuals),
                "skewness": stats.skew(residuals),
                "kurtosis": stats.kurtosis(residuals),
            }

            # Overall diagnostic summary
            passes_ljung_box = diagnostics["ljung_box"]["passes"]
            passes_normality = diagnostics["jarque_bera"]["passes"]

            diagnostics["overall_assessment"] = {
                "residuals_uncorrelated": passes_ljung_box,
                "residuals_normal": passes_normality,
                "model_adequate": passes_ljung_box and passes_normality,
            }

            print(
                f"  Ljung-Box test (residual autocorr): {'PASS' if passes_ljung_box else 'FAIL'}"
            )
            print(
                f"  Jarque-Bera test (normality): {'PASS' if passes_normality else 'FAIL'}"
            )
            print(
                f"  Overall model adequacy: {'ADEQUATE' if diagnostics['overall_assessment']['model_adequate'] else 'QUESTIONABLE'}"
            )

        except Exception as e:
            print(f"Diagnostic tests failed: {e}")
            diagnostics["error"] = str(e)

        return diagnostics

    def get_model_comparison(self):
        """
        Compare improved ARIMA and SARIMA models

        Returns:
        --------
        dict : Model comparison results
        """
        if "improved_arima" not in self.models or "improved_sarima" not in self.models:
            print("Both ARIMA and SARIMA models must be fitted for comparison")
            return None

        arima_results = self.models["improved_arima"]
        sarima_results = self.models["improved_sarima"]

        comparison: dict[str, Any] = {
            "arima": {
                "aic": arima_results["model_summary"]["aic"],
                "bic": arima_results["model_summary"]["bic"],
                "forecast_increase": arima_results["model_summary"][
                    "forecast_increase"
                ],
                "diagnostic_passes": arima_results["diagnostics"]["overall_assessment"][
                    "model_adequate"
                ],
            },
            "sarima": {
                "aic": sarima_results["model_summary"]["aic"],
                "bic": sarima_results["model_summary"]["bic"],
                "forecast_increase": sarima_results["model_summary"][
                    "forecast_increase"
                ],
                "diagnostic_passes": sarima_results["diagnostics"][
                    "overall_assessment"
                ]["model_adequate"],
            },
        }

        # Determine best model
        arima_aic = comparison["arima"]["aic"]
        sarima_aic = comparison["sarima"]["aic"]

        if sarima_aic < arima_aic - 2:  # Substantial improvement
            comparison["recommended"] = "sarima"
            comparison["reason"] = (
                f"SARIMA preferred (AIC difference: {arima_aic - sarima_aic:.2f})"
            )
        elif arima_aic < sarima_aic - 2:
            comparison["recommended"] = "arima"
            comparison["reason"] = (
                f"ARIMA preferred (AIC difference: {sarima_aic - arima_aic:.2f})"
            )
        else:
            comparison["recommended"] = "arima"  # Prefer simpler model
            comparison["reason"] = "Similar performance, prefer simpler ARIMA model"

        print("\\nModel Comparison:")
        print(
            f"  ARIMA  - AIC: {arima_aic:.2f}, Diagnostics: {'PASS' if comparison['arima']['diagnostic_passes'] else 'FAIL'}"
        )
        print(
            f"  SARIMA - AIC: {sarima_aic:.2f}, Diagnostics: {'PASS' if comparison['sarima']['diagnostic_passes'] else 'FAIL'}"
        )
        print(
            f"  Recommendation: {comparison['recommended'].upper()} ({comparison['reason']})"
        )

        return comparison


if __name__ == "__main__":
    print("Improved ARIMA/SARIMA Module")
    print("Enhanced time series modeling with proper stationarity testing")
    print("and comprehensive diagnostics for climate forecasting.")
