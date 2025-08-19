"""
Uncertainty Quantification Module
================================

Methods for quantifying prediction uncertainty in climate models including
confidence intervals, prediction intervals, and ensemble approaches.

"""

from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


class UncertaintyQuantifier:
    """
    Quantify and visualize prediction uncertainty for climate models
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize uncertainty quantifier

        Parameters:
        -----------
        confidence_level : float
            Confidence level for intervals (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.prediction_intervals = {}
        self.ensemble_results = {}

    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func=None,
        n_bootstrap: int = 1000,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Calculate bootstrap confidence intervals for a statistic

        Parameters:
        -----------
        data : np.ndarray
            Input data
        statistic_func : callable
            Function to calculate statistic (default: np.mean)
        n_bootstrap : int
            Number of bootstrap samples
        random_state : int
            Random seed for reproducibility

        Returns:
        --------
        Dict[str, Any] : Bootstrap confidence interval results
        """
        if statistic_func is None:
            statistic_func = np.mean

        np.random.seed(random_state)

        n_samples = len(data)
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)

        bootstrap_stats = np.array(bootstrap_stats)

        # Calculate confidence interval
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

        return {
            "statistic": float(statistic_func(data)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "bootstrap_std": float(np.std(bootstrap_stats)),
            "bootstrap_samples": bootstrap_stats,
        }

    def prediction_interval_regression(
        self, y_true: np.ndarray, y_pred: np.ndarray, X_test: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Calculate prediction intervals for regression models

        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        X_test : np.ndarray, optional
            Test features for leverage adjustment

        Returns:
        --------
        Dict[str, Any] : Prediction interval results
        """
        residuals = y_true - y_pred

        # Calculate residual standard error
        n = len(residuals)
        mse = mean_squared_error(y_true, y_pred)
        residual_std = np.sqrt(mse)

        # t-distribution critical value
        degrees_freedom = max(n - 2, 1)  # n - p (assuming 2 parameters)
        t_critical = stats.t.ppf(1 - self.alpha / 2, degrees_freedom)

        # Basic prediction interval (constant width)
        margin_error = t_critical * residual_std * np.sqrt(1 + 1 / n)

        pi_lower = y_pred - margin_error
        pi_upper = y_pred + margin_error

        # Calculate coverage probability
        coverage = np.mean((y_true >= pi_lower) & (y_true <= pi_upper))

        results = {
            "predictions": y_pred,
            "pi_lower": pi_lower,
            "pi_upper": pi_upper,
            "residual_std": float(residual_std),
            "margin_error": float(margin_error),
            "coverage_probability": float(coverage),
            "target_coverage": self.confidence_level,
            "degrees_freedom": degrees_freedom,
        }

        print("Prediction interval calculated:")
        print(f"  - Residual std: {residual_std:.4f}")
        print(f"  - Margin of error: {margin_error:.4f}")
        print(
            f"  - Coverage probability: {coverage:.3f} (target: {self.confidence_level:.3f})"
        )

        return results

    def quantile_regression_intervals(
        self, y_true: np.ndarray | None, predictions_list: list[np.ndarray]
    ) -> dict[str, Any]:
        """
        Calculate prediction intervals using quantile regression approach

        Parameters:
        -----------
        y_true : np.ndarray, optional
            True values (for validation)
        predictions_list : List[np.ndarray]
            List of prediction arrays from different models/bootstrap samples

        Returns:
        --------
        Dict[str, Any] : Quantile-based prediction intervals
        """
        predictions_array = np.column_stack(predictions_list)

        # Calculate quantiles across predictions
        lower_quantile = self.alpha / 2
        upper_quantile = 1 - (self.alpha / 2)

        pi_lower = np.quantile(predictions_array, lower_quantile, axis=1)
        pi_upper = np.quantile(predictions_array, upper_quantile, axis=1)
        mean_pred = np.mean(predictions_array, axis=1)
        std_pred = np.std(predictions_array, axis=1)

        # Calculate coverage if true values provided
        coverage = None
        if y_true is not None and len(y_true) == len(mean_pred):
            coverage = np.mean((y_true >= pi_lower) & (y_true <= pi_upper))

        results = {
            "mean_prediction": mean_pred,
            "prediction_std": std_pred,
            "pi_lower": pi_lower,
            "pi_upper": pi_upper,
            "coverage_probability": coverage,
            "n_models": len(predictions_list),
        }

        print(f"Quantile-based intervals from {len(predictions_list)} models:")
        print(f"  - Mean prediction std: {np.mean(std_pred):.4f}")
        if coverage is not None:
            print(f"  - Coverage probability: {coverage:.3f}")

        return results

    def time_series_prediction_intervals(
        self, residuals: np.ndarray, forecast: np.ndarray, forecast_steps: int
    ) -> dict[str, Any]:
        """
        Calculate prediction intervals for time series forecasts

        Parameters:
        -----------
        residuals : np.ndarray
            Model residuals from training
        forecast : np.ndarray
            Point forecasts
        forecast_steps : int
            Number of steps ahead

        Returns:
        --------
        Dict[str, Any] : Time series prediction intervals
        """
        # Residual standard error
        residual_std = np.std(residuals)

        # Variance increases with forecast horizon for most time series models
        # Use simple linear increase as approximation
        forecast_variance = residual_std**2 * (1 + np.arange(forecast_steps) * 0.1)
        forecast_std = np.sqrt(forecast_variance)

        # Normal distribution critical value
        z_critical = stats.norm.ppf(1 - self.alpha / 2)

        # Prediction intervals
        margin_error = z_critical * forecast_std
        pi_lower = forecast - margin_error
        pi_upper = forecast + margin_error

        results = {
            "forecast": forecast,
            "forecast_std": forecast_std,
            "pi_lower": pi_lower,
            "pi_upper": pi_upper,
            "residual_std": float(residual_std),
            "increasing_uncertainty": True,
        }

        print("Time series prediction intervals:")
        print(f"  - Base residual std: {residual_std:.4f}")
        print(f"  - Final forecast std: {forecast_std[-1]:.4f}")
        print(f"  - Uncertainty growth: {forecast_std[-1]/forecast_std[0]:.2f}x")

        return results

    def ensemble_uncertainty(
        self,
        model_predictions: dict[str, np.ndarray],
        model_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate ensemble prediction with uncertainty quantification

        Parameters:
        -----------
        model_predictions : Dict[str, np.ndarray]
            Dictionary of predictions from different models
        model_weights : Dict[str, float], optional
            Weights for each model (default: equal weights)

        Returns:
        --------
        Dict[str, Any] : Ensemble results with uncertainty
        """
        model_names = list(model_predictions.keys())
        n_models = len(model_names)

        if model_weights is None:
            model_weights = dict.fromkeys(model_names, 1.0 / n_models)

        # Normalize weights
        total_weight = sum(model_weights.values())
        model_weights = {k: v / total_weight for k, v in model_weights.items()}

        # Stack predictions
        predictions_matrix = np.column_stack(
            [model_predictions[name] for name in model_names]
        )
        weights_array = np.array([model_weights[name] for name in model_names])

        # Weighted ensemble mean
        ensemble_mean = np.dot(predictions_matrix, weights_array)

        # Model uncertainty (disagreement between models)
        model_variance = np.var(predictions_matrix, axis=1, ddof=1)

        # Within-model uncertainty (if available from individual models)
        # For now, use inter-model variance as uncertainty estimate
        ensemble_std = np.sqrt(model_variance)

        # Prediction intervals
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        pi_lower = ensemble_mean - z_critical * ensemble_std
        pi_upper = ensemble_mean + z_critical * ensemble_std

        results = {
            "ensemble_mean": ensemble_mean,
            "ensemble_std": ensemble_std,
            "model_variance": model_variance,
            "pi_lower": pi_lower,
            "pi_upper": pi_upper,
            "model_weights": model_weights,
            "individual_predictions": model_predictions,
        }

        print("Ensemble uncertainty quantification:")
        print(f"  - Models combined: {n_models}")
        print(f"  - Mean ensemble std: {np.mean(ensemble_std):.4f}")
        print(f"  - Model weights: {model_weights}")

        return results

    def extrapolation_uncertainty(
        self, training_range: tuple, prediction_range: tuple, base_uncertainty: float
    ) -> dict[str, Any]:
        """
        Quantify additional uncertainty from extrapolating beyond training data

        Parameters:
        -----------
        training_range : tuple
            Range of training data (min, max)
        prediction_range : tuple
            Range of prediction data (min, max)
        base_uncertainty : float
            Base model uncertainty within training range

        Returns:
        --------
        Dict[str, Any] : Extrapolation uncertainty results
        """
        train_min, train_max = training_range
        pred_min, pred_max = prediction_range

        # Calculate extrapolation distance
        extrapolation_distance = 0.0

        if pred_min < train_min:
            extrapolation_distance += train_min - pred_min
        if pred_max > train_max:
            extrapolation_distance += pred_max - train_max

        # Training range size
        training_span = train_max - train_min

        # Relative extrapolation distance
        relative_extrapolation = (
            extrapolation_distance / training_span if training_span > 0 else 0
        )

        # Increase uncertainty based on extrapolation distance
        # Simple linear increase - could be made more sophisticated
        extrapolation_multiplier = (
            1.0 + relative_extrapolation * 0.5
        )  # 50% increase per training span

        adjusted_uncertainty = base_uncertainty * extrapolation_multiplier

        results = {
            "base_uncertainty": float(base_uncertainty),
            "adjusted_uncertainty": float(adjusted_uncertainty),
            "extrapolation_distance": float(extrapolation_distance),
            "relative_extrapolation": float(relative_extrapolation),
            "uncertainty_multiplier": float(extrapolation_multiplier),
            "is_extrapolating": extrapolation_distance > 0,
        }

        if extrapolation_distance > 0:
            print("Extrapolation uncertainty adjustment:")
            print(f"  - Training range: {train_min:.1f} to {train_max:.1f}")
            print(f"  - Prediction range: {pred_min:.1f} to {pred_max:.1f}")
            print(f"  - Extrapolation distance: {extrapolation_distance:.2f}")
            print(f"  - Uncertainty multiplier: {extrapolation_multiplier:.2f}")
            print(f"  - Adjusted uncertainty: {adjusted_uncertainty:.4f}")
        else:
            print("No extrapolation detected - predictions within training range")

        return results

    def generate_uncertainty_report(self, all_results: dict[str, dict]) -> str:
        """
        Generate comprehensive uncertainty analysis report

        Parameters:
        -----------
        all_results : Dict[str, Dict]
            Dictionary of results from different uncertainty methods

        Returns:
        --------
        str : Formatted uncertainty report
        """
        report = "\\n" + "=" * 70 + "\\n"
        report += "UNCERTAINTY QUANTIFICATION REPORT\\n"
        report += "=" * 70 + "\\n"

        report += f"\\nConfidence Level: {self.confidence_level*100:.1f}%\\n"

        for method_name, results in all_results.items():
            report += f"\\n{method_name.upper().replace('_', ' ')}:\\n"
            report += "-" * 50 + "\\n"

            if (
                "coverage_probability" in results
                and results["coverage_probability"] is not None
            ):
                coverage = results["coverage_probability"]
                report += f"  - Coverage Probability: {coverage:.3f}\\n"

            if "residual_std" in results:
                report += (
                    f"  - Residual Standard Error: {results['residual_std']:.4f}\\n"
                )

            if "ensemble_std" in results:
                std_mean = np.mean(results["ensemble_std"])
                std_max = np.max(results["ensemble_std"])
                report += f"  - Mean Uncertainty: {std_mean:.4f}\\n"
                report += f"  - Maximum Uncertainty: {std_max:.4f}\\n"

            if "uncertainty_multiplier" in results:
                multiplier = results["uncertainty_multiplier"]
                report += f"  - Extrapolation Multiplier: {multiplier:.2f}\\n"

            if "n_models" in results:
                report += f"  - Models Combined: {results['n_models']}\\n"

        report += "\\n" + "=" * 70 + "\\n"
        report += "RECOMMENDATIONS:\\n"
        report += "- Use ensemble methods for robust predictions\\n"
        report += "- Account for extrapolation uncertainty in long-term forecasts\\n"
        report += "- Validate prediction intervals with out-of-sample data\\n"
        report += "- Consider model uncertainty in decision making\\n"
        report += "=" * 70

        return report


if __name__ == "__main__":
    print("Uncertainty Quantification Module")
    print("Methods for quantifying prediction uncertainty in climate models")
    print("including confidence intervals and ensemble approaches.")
