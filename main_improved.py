"""
Improved Heatwave Analysis Main Module
=====================================

Integrated pipeline using improved data validation, time series methods,
and uncertainty quantification for climate forecasting.

"""

# Import improved modules
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from data_loader import (
    combine_datasets,
    get_dataset_summary,
    load_deforestation_data,
    load_heatwave_data,
    print_data_summary,
)
from improved_arima import ImprovedTimeSeriesPredictor
from statistical_analysis import comprehensive_statistical_analysis, get_key_insights
from visualization import (
    create_summary_dashboard,
    plot_arima_decomposition_colorful,
    plot_sarima_enhanced,
    plot_time_series_results,
)

warnings.filterwarnings("ignore")

# Format helper functions for safe string formatting


def _fmt_int(x, default="Unknown"):
    """Return x with thousands separators if numeric, else a safe string."""
    if x is None:
        return default
    # numpy/py numbers
    if isinstance(x, int | np.integer):
        return f"{int(x):,}"
    # numeric-looking strings or floats → round to int for counts
    try:
        xi = int(float(x))
        return f"{xi:,}"
    except Exception:
        return str(x)


def _fmt_float(x, nd=2, default="Unknown"):
    """Format floats safely; falls back to string."""
    if x is None:
        return default
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return str(x)


# Import original modules for comparison
try:
    from stat_models import run_statistical_analysis

    STAT_MODELS_AVAILABLE = True
except ImportError:
    run_statistical_analysis = None  # type: ignore
    STAT_MODELS_AVAILABLE = False
    print("WARNING: stat_models.py not available")


# Try improved LSTM first (preferred for Python 3.13+)
print("WARNING: TensorFlow-based LSTM not available")

# Try PyTorch LSTM first (preferred for Python 3.13+)
try:
    from pytorch_lstm import PYTORCH_AVAILABLE, PyTorchLSTMPredictor

    if PYTORCH_AVAILABLE:
        print("INFO: PyTorch LSTM available for Python 3.13+ compatibility")
        PYTORCH_LSTM_AVAILABLE = True
    else:
        PYTORCH_LSTM_AVAILABLE = False
except ImportError:
    PyTorchLSTMPredictor = None  # type: ignore
    PYTORCH_AVAILABLE = False
    PYTORCH_LSTM_AVAILABLE = False
    print("WARNING: PyTorch-based LSTM not available")


class ImprovedHeatwaveAnalysis:
    """
    Main class for comprehensive heatwave analysis with improvements
    """

    def __init__(self):
        """Initialize the improved analysis pipeline"""
        # Initialize results at the start of the run
        self.results = {
            "data": {},
            "eda": {},
            "models": {},
            "counts": {},
            "ensemble": None,
            "uncertainty": None,
        }

        self.data = None
        self.primary_temp_col = None
        self.tree_loss_by_year = None
        self.combined_data = None
        self.annual_temp_stats = None
        self.statistical_results = None
        self.models = {}
        self.forecasts = {}

        print("=" * 70)
        print("IMPROVED HEATWAVE ANALYSIS PIPELINE")
        print("=" * 70)
        print("Enhanced with:")
        print("- Data validation and quality checks")
        print("- Proper time series validation")
        print("- Stationarity testing and differencing")
        print("- Uncertainty quantification")
        print("- Improved model architectures")
        print("=" * 70)

    def _dict_to_sorted_array(self, d: dict[str, float]) -> np.ndarray:
        """Convert dict to sorted numpy array by years"""
        if not d:
            return np.array([], dtype=float)
        years = sorted(d.keys())
        return np.array([float(d[y]) for y in years], dtype=float)

    def load_and_validate_data(self):
        """Load and validate all data with comprehensive checks"""
        print("\nStep 1: Loading and validating data...")

        try:
            # Load heatwave data with validation and get primary column
            self.data, self.primary_temp_col = load_heatwave_data()

            # Load deforestation data with validation (returns DataFrame, Series)
            deforestation_data, self.tree_loss_by_year = load_deforestation_data()

            # Combine datasets with primary temperature column (pass Series not DataFrame)
            self.combined_data, self.annual_temp_stats = combine_datasets(
                self.data, self.tree_loss_by_year, self.primary_temp_col
            )

            # Store core facts in results dict (single source of truth)
            start = str(self.data.index.min().date())
            end = str(self.data.index.max().date())
            self.results["data"].update(
                {
                    "primary_col": self.primary_temp_col,
                    "n_obs": int(self.data[self.primary_temp_col].notna().sum()),
                    "start": start,
                    "end": end,
                }
            )

            # Export span for loader context
            globals()["GLOBAL_CLIMATE_START_YEAR"] = int(self.data.index.min().year)
            globals()["GLOBAL_CLIMATE_END_YEAR"] = int(self.data.index.max().year)

            # Print comprehensive summary
            summary = get_dataset_summary(self.data)
            print_data_summary(summary)

            print("Data loading and validation completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Data loading failed: {e}")
            return False

    def generate_data_exploration_visualizations(self):
        """Generate comprehensive data exploration visualizations"""
        print("\nStep 1.5: Generating data exploration visualizations...")

        if self.data is None or self.primary_temp_col is None:
            print("INFO: data unavailable; skipping this step.")
            return False

        try:
            # Import visualization functions
            from plot_visuals import plot_annual_heatwaves, plot_monthly_trends
            from visualization import (
                plot_correlation_matrix,
                plot_deforestation_analysis,
                plot_heatwave_analysis,
                plot_temperature_trends,
            )

            # Generate monthly trends with canonical columns
            print("Creating monthly trends visualization...")
            plot_monthly_trends(
                self.data, self.primary_temp_col, "images/data_exploration"
            )

            # Generate annual heatwave visualization
            print("Creating annual heatwaves visualization...")
            if self.annual_temp_stats is not None:
                plot_annual_heatwaves(self.annual_temp_stats, "images/data_exploration")
            else:
                print(
                    "INFO: annual_temp_stats is None; skipping annual heatwaves visualization."
                )

            # Generate comprehensive temperature trends analysis
            print("Creating temperature trends analysis...")
            if self.annual_temp_stats is not None:
                plot_temperature_trends(
                    self.data,
                    self.annual_temp_stats,
                    self.primary_temp_col,
                    "images/data_exploration",
                )

            # Generate deforestation analysis if we have deforestation data
            print("Creating deforestation analysis...")
            if self.tree_loss_by_year is not None:
                plot_deforestation_analysis(
                    self.data,
                    self.tree_loss_by_year,
                    output_dir="images/data_exploration",
                )

            # Generate heatwave analysis
            print("Creating heatwave analysis...")
            plot_heatwave_analysis(
                self.data,
                heatwave_summary=self.annual_temp_stats,
                output_dir="images/data_exploration",
            )

            # Generate correlation matrix
            print("Creating correlation matrix...")
            if self.combined_data is not None:
                plot_correlation_matrix(
                    self.combined_data, output_dir="images/data_exploration"
                )

            # Store to results dict (single source of truth)
            self.results["eda"]["annual"] = self.annual_temp_stats

            print("Data exploration visualizations completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Data exploration visualization failed: {e}")
            return False

    def run_statistical_analysis(self):
        """Run comprehensive statistical analysis"""
        print("\nStep 2: Running statistical analysis...")

        if self.data is None or self.primary_temp_col is None:
            print("INFO: data unavailable; skipping this step.")
            return False

        try:
            # Run comprehensive statistical tests
            self.statistical_results = comprehensive_statistical_analysis(
                self.data,  # pyright: ignore[reportArgumentType]
                self.combined_data,  # pyright: ignore[reportArgumentType]
                (
                    self.tree_loss_by_year.to_frame()
                    if self.tree_loss_by_year is not None
                    else None
                ),
                self.annual_temp_stats,  # pyright: ignore[reportArgumentType]
            )

            # Store results in single source of truth
            self.results["eda"]["statistical"] = self.statistical_results

            # Extract key insights
            insights = get_key_insights(self.statistical_results)
            self.results["eda"]["insights"] = insights

            print("\nKEY STATISTICAL INSIGHTS:")
            for i, insight in enumerate(insights, 1):
                print(f"  {i}. {insight}")

            print("Statistical analysis completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Statistical analysis failed: {e}")
            return False

    def run_time_series_models(self):
        """Run improved ARIMA and SARIMA models"""
        print("\nStep 3: Running improved time series models...")

        if self.data is None or self.primary_temp_col is None:
            print("INFO: data unavailable; skipping this step.")
            return False

        try:
            # Initialize improved time series predictor
            ts_predictor = ImprovedTimeSeriesPredictor(
                self.data, self.tree_loss_by_year
            )

            # Fit improved ARIMA
            print("\nFitting improved ARIMA model...")
            arima_results = ts_predictor.fit_improved_arima()

            if arima_results:
                self.models["improved_arima"] = arima_results
                self.forecasts["arima"] = arima_results["annual_forecasts"]
                # Store in results dict (single source of truth)
                self.results["models"]["arima"] = arima_results
                # Store normalized for uncertainty
                annual_dict = arima_results.get("annual_forecasts", {})
                self.results["models"]["arima_norm"] = {
                    "yhat": self._dict_to_sorted_array(annual_dict),
                    "rmse": float(arima_results.get("val_rmse", np.inf)),
                    "name": "arima",
                }

            # Fit improved SARIMA
            print("\nFitting improved SARIMA model...")
            sarima_results = ts_predictor.fit_improved_sarima()

            if sarima_results:
                self.models["improved_sarima"] = sarima_results
                self.forecasts["sarima"] = sarima_results["annual_forecasts"]
                # Store in results dict (single source of truth)
                self.results["models"]["sarima"] = sarima_results
                # Store normalized for uncertainty
                annual_dict = sarima_results.get("annual_forecasts", {})
                self.results["models"]["sarima_norm"] = {
                    "yhat": self._dict_to_sorted_array(annual_dict),
                    "rmse": float(sarima_results.get("val_rmse", np.inf)),
                    "name": "sarima",
                }

            # Compare models
            if arima_results and sarima_results:
                comparison = ts_predictor.get_model_comparison()
                if comparison and "recommended" in comparison:
                    print(
                        f"\nModel comparison completed: {comparison['recommended']} recommended"
                    )

            # Generate time series visualizations
            print("\nGenerating time series visualizations...")
            if self.forecasts:
                plot_time_series_results(self.forecasts)

                # Generate individual model visualizations
                if (
                    "arima" in self.models
                    and "decomposition" in self.models["improved_arima"]
                ):
                    plot_arima_decomposition_colorful(
                        self.models["improved_arima"]["decomposition"],
                        output_dir="images/arima",
                    )

                if "sarima" in self.forecasts:
                    plot_sarima_enhanced(self.forecasts, output_dir="images/sarima")

            print("Time series modeling completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Time series modeling failed: {e}")
            return False

    def run_statistical_count_models(self):
        """Run statistical count models (Poisson/Negative Binomial)"""
        print("\nStep 4: Running statistical count models...")

        if self.data is None or self.primary_temp_col is None:
            print("INFO: data unavailable; skipping this step.")
            return False

        if not STAT_MODELS_AVAILABLE:
            print("Statistical count models not available - skipping")
            return True

        try:
            # Run count-based statistical analysis
            if run_statistical_analysis is not None:
                stat_predictor = run_statistical_analysis(self.data)
            else:
                print("run_statistical_analysis is not available")
                return True

            if stat_predictor and hasattr(stat_predictor, "predictions"):
                self.models["statistical"] = stat_predictor
                # Store in results dict (single source of truth)
                self.results["counts"]["statistical"] = stat_predictor

                # Extract predictions if available (with safe None check)
                if (
                    stat_predictor.predictions
                    and "poisson" in stat_predictor.predictions
                ):
                    self.forecasts["poisson"] = stat_predictor.predictions["poisson"]
                    self.results["counts"]["poisson"] = stat_predictor.predictions[
                        "poisson"
                    ]
                if (
                    stat_predictor.predictions
                    and "negative_binomial" in stat_predictor.predictions
                ):
                    self.forecasts["negative_binomial"] = stat_predictor.predictions[
                        "negative_binomial"
                    ]
                    self.results["counts"]["negative_binomial"] = (
                        stat_predictor.predictions["negative_binomial"]
                    )

            print("Statistical count modeling completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Statistical count modeling failed: {e}")
            return False

    def run_lstm_model(self):
        """Run LSTM model with PyTorch preference"""
        print("\nStep 5: Running LSTM model (PyTorch preferred)...")

        if self.data is None or self.primary_temp_col is None:
            print("INFO: data unavailable; skipping this step.")
            return False

        try:
            # Add robust LSTM guard
            import numpy as np

            from data_loader import _ensure_datetime_index

            # Ensure datetime index exists
            if self.data is not None:
                self.data = _ensure_datetime_index(self.data)

            # Attach deforestation feature safely
            if (
                self.tree_loss_by_year is not None
                and len(self.tree_loss_by_year.dropna()) > 0
            ):
                ymap = {
                    int(k): float(v) for k, v in self.tree_loss_by_year.dropna().items()  # type: ignore
                }
                years = pd.DatetimeIndex(self.data.index).year.astype(int)  # type: ignore
                self.data["tree_loss_ha"] = [ymap.get(y, 0.0) for y in years]
            else:
                self.data["tree_loss_ha"] = 0.0

            # Build features
            idx = self.data.index
            doy = idx.dayofyear.to_numpy()  # type: ignore
            sin_doy = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
            cos_doy = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
            yarr = self.data[self.primary_temp_col].astype(float).to_numpy(np.float32)
            tree = self.data["tree_loss_ha"].astype(float).to_numpy(np.float32)
            Xmat = np.column_stack([yarr, sin_doy, cos_doy, tree]).astype(np.float32)

            # If too short, skip gracefully
            if Xmat.shape[0] < 400:
                print("INFO: Not enough data for LSTM; skipping.")
                return True
            # First try PyTorch LSTM
            try:
                from pytorch_lstm import PYTORCH_AVAILABLE, PyTorchLSTMPredictor

                if PYTORCH_AVAILABLE:
                    print("Using PyTorch LSTM implementation...")

                    # Initialize PyTorch LSTM predictor
                    lstm_predictor = PyTorchLSTMPredictor(
                        self.data, self.tree_loss_by_year
                    )

                    # Fit PyTorch LSTM
                    lstm_results = lstm_predictor.fit_pytorch_lstm()

                    if lstm_results:
                        self.models["pytorch_lstm"] = lstm_results
                        self.forecasts["lstm"] = lstm_results["annual_forecasts"]
                        # Store in results dict (single source of truth)
                        self.results["models"]["pytorch_lstm"] = lstm_results
                        # NEW: normalized entry for uncertainty step
                        self.results["models"]["pytorch_lstm_norm"] = {
                            "yhat": np.array(
                                list(lstm_results["annual_forecasts"].values()),
                                dtype=float,
                            ),
                            "rmse": float(
                                lstm_results.get("validation_metrics", {}).get(
                                    "rmse", np.inf
                                )
                            ),
                            "name": "pytorch_lstm",
                        }

                        print("PyTorch LSTM modeling completed successfully!")
                        return True

            except ImportError:
                print("PyTorch LSTM not available, trying TensorFlow fallback...")

        except Exception as e:
            print(f"ERROR: LSTM modeling failed: {e}")
            return False

    def generate_model_visualizations(self):
        """Generate machine learning and model comparison visualizations"""
        print("\nStep 5.5: Generating model performance visualizations...")

        try:
            # Import plot_ml_results function
            from visualization import plot_ml_results

            # Generate ML model results if we have ML forecasts
            ml_forecasts = {}
            for key in ["random_forest", "xgboost", "lstm"]:
                if key in self.forecasts:
                    ml_forecasts[key] = self.forecasts[key]

            if ml_forecasts:
                print("Creating ML model performance visualizations...")
                plot_ml_results(ml_forecasts)

            print("Model visualization generation completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Model visualization generation failed: {e}")
            return False

    def quantify_uncertainty(self):
        """Quantify prediction uncertainty across all models"""
        print("\nStep 6: Quantifying prediction uncertainty...")
        bag = self.results.get("models", {})
        ok = {}
        for k, v in bag.items():
            if not isinstance(v, dict):
                continue
            yhat = np.asarray(v.get("yhat", []), float)
            rmse = float(v.get("rmse", np.inf))
            if yhat.size == 0 or not np.isfinite(yhat).all():
                continue
            if not np.isfinite(rmse) or rmse <= 0:
                continue
            ok[k] = {"yhat": yhat, "rmse": rmse}

        if not ok:
            print("INFO: No valid models for uncertainty; skipping.")
            self.results["ensemble"] = None
            self.results["uncertainty"] = None
            return False

        w = np.array([1.0 / v["rmse"] for v in ok.values()], float)
        w = w / w.sum()
        Y = np.stack([v["yhat"] for v in ok.values()], axis=0)

        ens = np.tensordot(w, Y, axes=(0, 0))
        dif = Y - ens
        var = np.tensordot(w, dif**2, axes=(0, 0))
        std = np.sqrt(var)

        self.results["ensemble"] = ens
        self.results["uncertainty"] = std
        print("Uncertainty quantification completed successfully!")
        return True

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report with forecasts"""
        print("\nStep 7: Generating comprehensive report and summary dashboard...")

        # Generate the comprehensive summary dashboard
        try:
            print("Creating comprehensive summary dashboard...")
            create_summary_dashboard(
                self.data,
                self.statistical_results,
                self.tree_loss_by_year,
                output_dir="images/summary",
            )
            print("✓ Summary dashboard created successfully!")
        except Exception as e:
            print(f"WARNING: Summary dashboard creation failed: {e}")

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE HEATWAVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Data Summary
        if "data" in self.results:
            data_info = self.results["data"]
            report.append("DATA SUMMARY")
            report.append("-" * 30)
            report.append(f"Dataset: {data_info.get('source', 'Unknown')}")
            report.append(f"Records: {_fmt_int(data_info.get('total_records'))}")
            report.append(f"Date Range: {data_info.get('date_range', 'Unknown')}")
            report.append(
                f"Primary Temperature Column: {data_info.get('primary_col', 'Unknown')}"
            )
            if self.data is not None:
                temp_stats = self.data[data_info["primary_col"]].describe()
                report.append("Temperature Statistics:")
                report.append(f"  Mean: {temp_stats['mean']:.2f}°C")
                report.append(f"  Std:  {temp_stats['std']:.2f}°C")
                report.append(f"  Min:  {temp_stats['min']:.2f}°C")
                report.append(f"  Max:  {temp_stats['max']:.2f}°C")
            report.append("")

        # Statistical Analysis
        if "statistics" in self.results:
            stats = self.results["statistics"]
            report.append("STATISTICAL ANALYSIS")
            report.append("-" * 30)
            if "trends" in stats:
                trends = stats["trends"]
                report.append("Temperature Trends:")
                report.append(f"  Slope: {trends.get('slope', 'N/A'):.6f}°C/day")
                report.append(
                    f"  Annual Change: {trends.get('annual_change', 'N/A'):.4f}°C/year"
                )
                report.append(f"  P-value: {trends.get('p_value', 'N/A'):.2e}")
                report.append(f"  Trend: {trends.get('trend_significance', 'Unknown')}")

            if "extreme_analysis" in stats:
                extreme = stats["extreme_analysis"]
                report.append("Extreme Temperature Analysis:")
                report.append(f"  Heat Days (>35°C): {extreme.get('heat_days', 'N/A')}")
                report.append(
                    f"  Very Hot Days (>40°C): {extreme.get('very_hot_days', 'N/A')}"
                )
                report.append(
                    f"  Record High: {extreme.get('record_high', 'N/A'):.2f}°C"
                )
                report.append(f"  Record Date: {extreme.get('record_date', 'N/A')}")
            report.append("")

        # Heatwave Analysis
        if "heatwaves" in self.results:
            hw_stats = self.results["heatwaves"]
            report.append("HEATWAVE ANALYSIS")
            report.append("-" * 30)
            report.append(f"Total Heatwaves: {hw_stats.get('total_heatwaves', 'N/A')}")
            report.append(
                f"Average Duration: {hw_stats.get('avg_duration', 'N/A'):.1f} days"
            )
            report.append(
                f"Maximum Duration: {hw_stats.get('max_duration', 'N/A')} days"
            )
            report.append(
                f"Average Intensity: {hw_stats.get('avg_intensity', 'N/A'):.2f}°C"
            )
            report.append(
                f"Maximum Intensity: {hw_stats.get('max_intensity', 'N/A'):.2f}°C"
            )

            if "yearly_counts" in hw_stats:
                recent_years = list(hw_stats["yearly_counts"].keys())[-10:]
                report.append("Recent Heatwave Frequency (last 10 years):")
                for year in recent_years:
                    count = hw_stats["yearly_counts"][year]
                    report.append(f"  {year}: {count} heatwaves")
            report.append("")

        # Model Results
        if "models" in self.results:
            models = self.results["models"]
            report.append("MODEL PERFORMANCE")
            report.append("-" * 30)

            # ARIMA/SARIMA Results
            for model_name in ["arima", "sarima"]:
                if model_name in models:
                    model_data = models[model_name]
                    report.append(f"{model_name.upper()} Model:")
                    if "aic" in model_data:
                        report.append(f"  AIC: {model_data['aic']:.2f}")
                    if "bic" in model_data:
                        report.append(f"  BIC: {model_data['bic']:.2f}")
                    if "rmse" in model_data:
                        report.append(f"  RMSE: {model_data['rmse']:.4f}°C")
                    if "mae" in model_data:
                        report.append(f"  MAE: {model_data['mae']:.4f}°C")
                    report.append("")

            # ML Model Results
            ml_models = ["random_forest", "xgboost", "pytorch_lstm"]
            for model_name in ml_models:
                if model_name in models:
                    model_data = models[model_name]
                    if "validation_metrics" in model_data:
                        metrics = model_data["validation_metrics"]
                        report.append(f"{model_name.replace('_', ' ').title()} Model:")
                        report.append(f"  R²: {metrics.get('r2', 'N/A'):.4f}")
                        report.append(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}°C")
                        report.append(f"  MAE: {metrics.get('mae', 'N/A'):.4f}°C")
                        report.append("")

        # Temperature Forecasts
        if self.forecasts:
            report.append("TEMPERATURE FORECASTS (2025-2030)")
            report.append("-" * 40)

            # Individual model forecasts
            for model_name, predictions in self.forecasts.items():
                if isinstance(predictions, dict) and predictions:
                    report.append(f"{model_name.upper()} Forecasts:")
                    for year, temp in sorted(predictions.items()):
                        if isinstance(year, str | int) and isinstance(
                            temp, int | float
                        ):
                            report.append(f"  {year}: {temp:.2f}°C")
                    report.append("")

            # Ensemble average if multiple models
            if len(self.forecasts) > 1:
                # Compute ensemble average
                all_years = set()
                for predictions in self.forecasts.values():
                    if isinstance(predictions, dict):
                        all_years.update(predictions.keys())

                ensemble_avg = {}
                for year in sorted(all_years):
                    temps = []
                    for predictions in self.forecasts.values():
                        if (
                            isinstance(predictions, dict)
                            and year in predictions
                            and isinstance(predictions[year], int | float)
                        ):
                            temps.append(predictions[year])
                    if temps:
                        ensemble_avg[year] = np.mean(temps)

                if ensemble_avg:
                    report.append("  Ensemble Average:")
                    # Replace historical_avg logic with canonical baseline
                    if self.data is not None:
                        baseline_col = (
                            "temp_mean_c"
                            if "temp_mean_c" in self.data.columns
                            else self.primary_temp_col
                        )
                        historical_avg = float(self.data[baseline_col].mean())
                    else:
                        historical_avg = 30.0

                    for year, temp in ensemble_avg.items():
                        increase = temp - historical_avg
                        report.append(
                            f"    {year}: {temp:.2f}°C ({increase:+.2f}°C from historical)"
                        )
                    report.append("")

        # Uncertainty Quantification
        if "uncertainty" in self.results and self.results["uncertainty"] is not None:
            unc_std = self.results["uncertainty"]  # This is a numpy array
            ensemble = self.results.get("ensemble", None)  # This is a numpy array
            report.append("UNCERTAINTY QUANTIFICATION")
            report.append("-" * 30)

            # Count valid models from results
            valid_models = []
            for k, v in self.results.get("models", {}).items():
                if isinstance(v, dict) and "yhat" in v:
                    valid_models.append(k.replace("_norm", ""))
            report.append(f"Models Combined: {len(valid_models)}")

            if isinstance(unc_std, np.ndarray) and unc_std.size > 0:
                avg_uncertainty = float(np.mean(unc_std))
                report.append(
                    f"Average Model Uncertainty (σ): ±{avg_uncertainty:.3f}°C"
                )

                # If we have ensemble predictions, show intervals for first prediction
                if isinstance(ensemble, np.ndarray) and ensemble.size > 0:
                    first_pred = float(ensemble[0])
                    first_std = float(unc_std[0])
                    lower = first_pred - 1.96 * first_std
                    upper = first_pred + 1.96 * first_std
                    report.append("95% Prediction Interval for 2025:")
                    report.append(f"  Lower: {lower:.2f}°C")
                    report.append(f"  Upper: {upper:.2f}°C")
            report.append("")

        # Key Findings Summary
        report.append("KEY FINDINGS")
        report.append("-" * 30)

        # Temperature trends
        if "statistics" in self.results and "trends" in self.results["statistics"]:
            trends = self.results["statistics"]["trends"]
            if "annual_change" in trends:
                annual_change = trends["annual_change"]
                if annual_change > 0:
                    report.append(
                        f"• Significant warming trend: +{annual_change:.3f}°C/year"
                    )
                else:
                    report.append(f"• Temperature trend: {annual_change:.3f}°C/year")

        # Heatwave trends
        if "heatwaves" in self.results:
            hw_stats = self.results["heatwaves"]
            total_hw = hw_stats.get("total_heatwaves", 0)
            avg_intensity = hw_stats.get("avg_intensity", 0)
            report.append(f"• Historical heatwaves: {total_hw} events")
            if avg_intensity:
                report.append(
                    f"• Average heatwave intensity: {avg_intensity:.2f}°C above threshold"
                )

        # Future projections
        if self.forecasts:
            # Get representative future temperature
            future_temps = []
            for predictions in self.forecasts.values():
                if isinstance(predictions, dict):
                    future_temps.extend(
                        [v for v in predictions.values() if isinstance(v, int | float)]
                    )

            if future_temps:
                avg_future = np.mean(future_temps)
                if self.data is not None and hasattr(self, "primary_temp_col"):
                    current_avg = self.data[self.primary_temp_col].mean()
                    change = avg_future - current_avg
                    report.append(
                        f"• Projected temperature change (2025-2030): {change:+.2f}°C"
                    )
                else:
                    report.append(
                        f"• Average projected temperature (2025-2030): {avg_future:.2f}°C"
                    )

        report.append("")
        report.append("=" * 80)
        report.append("Analysis completed successfully.")
        report.append("=" * 80)

        # Join all lines and return
        full_report = "\n".join(report)

        # Save to file
        with open("heatwave_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(full_report)

        print("📄 Comprehensive report saved: heatwave_analysis_report.txt")
        return full_report

    def run_complete_analysis(self):
        """Run the complete improved analysis pipeline with visualizations"""
        print("Starting comprehensive heatwave analysis...")

        success_count = 0
        total_steps = 9  # Updated to include visualization steps

        # Step 1: Load and validate data
        ok = self.load_and_validate_data()
        if not ok:
            print("\nFatal: data load/validation failed — aborting remaining steps.")
            return False
        success_count += 1

        # Sanity asserts (cheap & powerful)
        assert isinstance(
            self.data.index,  # pyright: ignore[reportOptionalMemberAccess]
            pd.DatetimeIndex,
        )
        assert (
            self.primary_temp_col
            in self.data.columns  # pyright: ignore[reportOptionalMemberAccess]
        )
        assert (
            "is_heatwave_day"
            in self.data.columns  # pyright: ignore[reportOptionalMemberAccess]
        )

        # Step 1.5: Generate data exploration visualizations
        if self.generate_data_exploration_visualizations():
            success_count += 1

        # Step 2: Statistical analysis
        if self.run_statistical_analysis():
            success_count += 1

        # Step 3: Time series models (includes time series visualizations)
        if self.run_time_series_models():
            success_count += 1

        # Step 4: Statistical count models
        if self.run_statistical_count_models():
            success_count += 1

        # Step 5: LSTM model
        if self.run_lstm_model():
            success_count += 1

        # Step 5.5: Generate model performance visualizations
        if self.generate_model_visualizations():
            success_count += 1

        # Step 6: Uncertainty quantification
        if self.quantify_uncertainty():
            success_count += 1

        # Step 7: Generate comprehensive report (includes summary dashboard)
        if self.generate_comprehensive_report():
            success_count += 1

        # Final summary
        print("\n" + "=" * 70)
        print("ANALYSIS PIPELINE COMPLETED")
        print("=" * 70)
        print(f"Steps completed successfully: {success_count}/{total_steps}")
        print(f"Models fitted: {len(self.models)}")
        print(f"Forecasts generated: {len(self.forecasts)}")

        if success_count == total_steps:
            print("ALL STEPS COMPLETED SUCCESSFULLY!")
        else:
            print("Some steps had issues - check logs above")

        print("=" * 70)

        return success_count == total_steps


def main():
    """Main function to run the improved analysis"""

    # Create analysis instance
    analysis = ImprovedHeatwaveAnalysis()

    # Run complete analysis
    success = analysis.run_complete_analysis()

    if success:
        print("\nImproved heatwave analysis completed successfully!")
        print("Check 'heatwave_analysis_report.txt' for detailed results.")
    else:
        print("\nAnalysis completed with some issues.")
        print("Check the output above for specific errors.")

    return analysis


if __name__ == "__main__":
    main()
