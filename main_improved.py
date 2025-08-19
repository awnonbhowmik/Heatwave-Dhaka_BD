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

from data_loader import (
    combine_datasets,
    get_dataset_summary,
    load_deforestation_data,
    load_heatwave_data,
    print_data_summary,
)
from improved_arima import ImprovedTimeSeriesPredictor
from statistical_analysis import comprehensive_statistical_analysis, get_key_insights
from uncertainty_quantification import UncertaintyQuantifier

warnings.filterwarnings("ignore")

# Import original modules for comparison
try:
    from stat_models import run_statistical_analysis

    STAT_MODELS_AVAILABLE = True
except ImportError:
    run_statistical_analysis = None  # type: ignore
    STAT_MODELS_AVAILABLE = False
    print("WARNING: stat_models.py not available")

try:
    from improved_lstm import ImprovedLSTMPredictor

    IMPROVED_LSTM_AVAILABLE = True
except ImportError:
    ImprovedLSTMPredictor = None  # type: ignore
    IMPROVED_LSTM_AVAILABLE = False
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
        self.data = None
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

    def load_and_validate_data(self):
        """Load and validate all data with comprehensive checks"""
        print("\\nStep 1: Loading and validating data...")

        try:
            # Load heatwave data with validation
            self.data, threshold = load_heatwave_data()

            # Load deforestation data with validation
            deforestation_data, self.tree_loss_by_year = load_deforestation_data()

            # Combine datasets
            self.combined_data, self.annual_temp_stats = combine_datasets(
                self.data, self.tree_loss_by_year
            )

            # Print comprehensive summary
            summary = get_dataset_summary(self.data)
            print_data_summary(summary)

            print("Data loading and validation completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Data loading failed: {e}")
            return False

    def run_statistical_analysis(self):
        """Run comprehensive statistical analysis"""
        print("\\nStep 2: Running statistical analysis...")

        try:
            # Run comprehensive statistical tests
            self.statistical_results = comprehensive_statistical_analysis(
                self.data,
                self.combined_data,
                self.tree_loss_by_year,
                self.annual_temp_stats,
            )

            # Extract key insights
            insights = get_key_insights(self.statistical_results)

            print("\\nKEY STATISTICAL INSIGHTS:")
            for i, insight in enumerate(insights, 1):
                print(f"  {i}. {insight}")

            print("Statistical analysis completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Statistical analysis failed: {e}")
            return False

    def run_time_series_models(self):
        """Run improved ARIMA and SARIMA models"""
        print("\\nStep 3: Running improved time series models...")

        try:
            # Initialize improved time series predictor
            ts_predictor = ImprovedTimeSeriesPredictor(
                self.data, self.tree_loss_by_year
            )

            # Fit improved ARIMA
            print("\\nFitting improved ARIMA model...")
            arima_results = ts_predictor.fit_improved_arima()

            if arima_results:
                self.models["improved_arima"] = arima_results
                self.forecasts["arima"] = arima_results["annual_forecasts"]

            # Fit improved SARIMA
            print("\\nFitting improved SARIMA model...")
            sarima_results = ts_predictor.fit_improved_sarima()

            if sarima_results:
                self.models["improved_sarima"] = sarima_results
                self.forecasts["sarima"] = sarima_results["annual_forecasts"]

            # Compare models
            if arima_results and sarima_results:
                comparison = ts_predictor.get_model_comparison()
                if comparison and "recommended" in comparison:
                    print(
                        f"\\nModel comparison completed: {comparison['recommended']} recommended"
                    )

            print("Time series modeling completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Time series modeling failed: {e}")
            return False

    def run_statistical_count_models(self):
        """Run statistical count models (Poisson/Negative Binomial)"""
        print("\\nStep 4: Running statistical count models...")

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

                # Extract predictions if available
                if "poisson" in stat_predictor.predictions:
                    self.forecasts["poisson"] = stat_predictor.predictions["poisson"]
                if "negative_binomial" in stat_predictor.predictions:
                    self.forecasts["negative_binomial"] = stat_predictor.predictions[
                        "negative_binomial"
                    ]

            print("Statistical count modeling completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Statistical count modeling failed: {e}")
            return False

    def run_lstm_model(self):
        """Run LSTM model with PyTorch preference"""
        print("\\nStep 5: Running LSTM model (PyTorch preferred)...")

        try:
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

                        print("PyTorch LSTM modeling completed successfully!")
                        return True

            except ImportError:
                print("PyTorch LSTM not available, trying TensorFlow fallback...")

            # Fallback to TensorFlow LSTM if available
            if IMPROVED_LSTM_AVAILABLE and ImprovedLSTMPredictor is not None:
                print("Using TensorFlow LSTM fallback...")

                # Initialize improved LSTM predictor
                lstm_predictor = ImprovedLSTMPredictor(
                    self.data, self.tree_loss_by_year
                )

                # Fit improved LSTM
                lstm_results = lstm_predictor.fit_improved_lstm()

                if lstm_results:
                    self.models["tensorflow_lstm"] = lstm_results
                    self.forecasts["lstm"] = lstm_results["annual_forecasts"]

                print("TensorFlow LSTM modeling completed successfully!")
                return True
            else:
                print(
                    "LSTM model not available (neither PyTorch nor TensorFlow) - skipping"
                )
                return True

        except Exception as e:
            print(f"ERROR: LSTM modeling failed: {e}")
            return False

    def quantify_uncertainty(self):
        """Quantify prediction uncertainty across all models"""
        print("\\nStep 6: Quantifying prediction uncertainty...")

        try:
            uncertainty_quantifier = UncertaintyQuantifier(confidence_level=0.95)

            # Collect model predictions for ensemble analysis
            model_predictions = {}

            for model_name, forecasts in self.forecasts.items():
                if isinstance(forecasts, dict):
                    # Extract annual forecasts as array
                    years = sorted(forecasts.keys())
                    predictions = [forecasts[year] for year in years]
                    model_predictions[model_name] = np.array(predictions)

            if len(model_predictions) >= 2:
                print(
                    f"Analyzing uncertainty across {len(model_predictions)} models..."
                )

                # Ensemble uncertainty analysis
                ensemble_results = uncertainty_quantifier.ensemble_uncertainty(
                    model_predictions
                )

                # Store ensemble results
                self.models["ensemble"] = ensemble_results

                # Extrapolation uncertainty (predicting 2025-2030 from 1972-2024 data)
                training_range = (1972, 2024)
                prediction_range = (2025, 2030)
                base_uncertainty = np.mean(ensemble_results["ensemble_std"])

                extrapolation_results = (
                    uncertainty_quantifier.extrapolation_uncertainty(
                        training_range, prediction_range, base_uncertainty
                    )
                )

                self.models["extrapolation_uncertainty"] = extrapolation_results

                print("Uncertainty quantification completed successfully!")

                # Print uncertainty summary
                print("\\nUNCERTAINTY ANALYSIS SUMMARY:")
                print(f"  - Ensemble models: {len(model_predictions)}")
                print(f"  - Base uncertainty: ±{base_uncertainty:.3f}°C")
                print(
                    f"  - Extrapolation multiplier: {extrapolation_results['uncertainty_multiplier']:.2f}"
                )
                print(
                    f"  - Adjusted uncertainty: ±{extrapolation_results['adjusted_uncertainty']:.3f}°C"
                )

            else:
                print("Not enough models for uncertainty analysis")

            return True

        except Exception as e:
            print(f"ERROR: Uncertainty quantification failed: {e}")
            return False

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\\nStep 7: Generating comprehensive report...")

        report = "\\n" + "=" * 80 + "\\n"
        report += "COMPREHENSIVE HEATWAVE ANALYSIS REPORT\\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n"
        report += "=" * 80 + "\\n"

        # Initialize temp_col with default
        temp_col = None
        if self.data is not None and len(self.data.columns) > 1:
            temp_col = self.data.columns[1]  # First temp column after timestamp

        # Data summary
        if self.data is not None:
            report += "\\nDATA SUMMARY:\\n"
            report += f"  - Records: {len(self.data):,} daily observations\\n"
            report += f"  - Period: {self.data['timestamp'].min().year}-{self.data['timestamp'].max().year}\\n"
            report += f"  - Heatwave days: {self.data['Heatwave'].sum():,} ({self.data['Heatwave'].mean()*100:.1f}% of days)\\n"

            if temp_col:
                report += f"  - Mean temperature: {self.data[temp_col].mean():.2f}°C\\n"
                report += f"  - Temperature range: {self.data[temp_col].min():.1f}°C to {self.data[temp_col].max():.1f}°C\\n"

        # Statistical findings
        if self.statistical_results:
            report += "\\nSTATISTICAL FINDINGS:\\n"

            if "temperature_trend" in self.statistical_results:
                trend = self.statistical_results["temperature_trend"]
                report += f"  - Temperature trend: +{trend['slope']:.4f}°C/year\\n"
                report += f"  - Total warming (1972-2024): +{trend['total_increase_52years']:.2f}°C\\n"
                report += f"  - Trend significance: {'Yes' if trend['is_significant'] else 'No'}\\n"

            if "stationarity" in self.statistical_results:
                stationarity = self.statistical_results["stationarity"]
                report += f"  - Time series stationarity: {'Yes' if stationarity['is_stationary'] else 'No'}\\n"

        # Model forecasts
        if self.forecasts:
            report += "\\nMODEL FORECASTS (2025-2030):\\n"

            # Calculate ensemble average if multiple models
            if len(self.forecasts) > 1:
                years = ["2025", "2026", "2027", "2028", "2029", "2030"]
                ensemble_avg = {}

                for year in years:
                    temps = []
                    for _model_name, forecasts in self.forecasts.items():
                        if year in forecasts:
                            temps.append(forecasts[year])

                    if temps:
                        ensemble_avg[year] = np.mean(temps)

                report += "  Ensemble Average:\\n"
                historical_avg = (
                    self.data[temp_col].mean()
                    if (self.data is not None and temp_col)
                    else 30.0
                )
                for year, temp in ensemble_avg.items():
                    increase = temp - historical_avg
                    report += f"    {year}: {temp:.2f}°C ({increase:+.2f}°C)\\n"

            # Individual model forecasts
            for model_name, forecasts in self.forecasts.items():
                report += f"  {model_name.title()} Model:\\n"
                for year, temp in forecasts.items():
                    report += f"    {year}: {temp:.2f}°C\\n"

        # Uncertainty analysis
        if "ensemble" in self.models:
            ensemble = self.models["ensemble"]
            report += "\\nUNCERTAINTY ANALYSIS:\\n"
            report += (
                f"  - Models combined: {len(ensemble['individual_predictions'])}\\n"
            )
            report += f"  - Average uncertainty: ±{np.mean(ensemble['ensemble_std']):.3f}°C\\n"

            if "extrapolation_uncertainty" in self.models:
                extrap = self.models["extrapolation_uncertainty"]
                report += f"  - Extrapolation adjustment: {extrap['uncertainty_multiplier']:.2f}x\\n"
                report += (
                    f"  - Final uncertainty: ±{extrap['adjusted_uncertainty']:.3f}°C\\n"
                )

        # Recommendations
        report += "\\nRECOMMENDATIONS:\\n"
        report += "  - Use ensemble forecasts for robust predictions\\n"
        report += "  - Account for increasing uncertainty in long-term forecasts\\n"
        report += "  - Consider model limitations and data gaps\\n"
        report += "  - Regularly update models with new data\\n"
        report += "  - Validate predictions with independent data\\n"

        # Limitations
        report += "\\nLIMITATIONS:\\n"
        report += "  - Deforestation data missing for 1972-2000 and 2024-2030\\n"
        report += "  - Models assume historical patterns continue\\n"
        report += "  - No consideration of extreme climate events\\n"
        report += "  - Urban heat island effects not explicitly modeled\\n"

        report += "\\n" + "=" * 80

        print(report)

        # Save report to file
        try:
            with open("heatwave_analysis_report.txt", "w") as f:
                f.write(report)
            print("\\nReport saved to: heatwave_analysis_report.txt")
        except Exception as e:
            print(f"Could not save report: {e}")

        return report

    def run_complete_analysis(self):
        """Run the complete improved analysis pipeline"""
        print("Starting comprehensive heatwave analysis...")

        success_count = 0
        total_steps = 7

        # Step 1: Load and validate data
        if self.load_and_validate_data():
            success_count += 1

        # Step 2: Statistical analysis
        if self.run_statistical_analysis():
            success_count += 1

        # Step 3: Time series models
        if self.run_time_series_models():
            success_count += 1

        # Step 4: Statistical count models
        if self.run_statistical_count_models():
            success_count += 1

        # Step 5: LSTM model
        if self.run_lstm_model():
            success_count += 1

        # Step 6: Uncertainty quantification
        if self.quantify_uncertainty():
            success_count += 1

        # Step 7: Generate report
        if self.generate_comprehensive_report():
            success_count += 1

        # Final summary
        print("\\n" + "=" * 70)
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
        print("\\nImproved heatwave analysis completed successfully!")
        print("Check 'heatwave_analysis_report.txt' for detailed results.")
    else:
        print("\\nAnalysis completed with some issues.")
        print("Check the output above for specific errors.")

    return analysis


if __name__ == "__main__":
    main()
