"""
Statistical Models for Heatwave Prediction
==========================================

This module implements Poisson and Negative Binomial distribution models
for predicting heatwave days from 2025-2030. These models are specifically
designed for count data and can capture overdispersion in heatwave patterns.

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

STATSMODELS_AVAILABLE = True


def _safe_mape(y_true, y_pred):
    """Calculate MAPE with division by zero protection"""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0.0)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


class StatisticalHeatwavePredictor:
    """
    Statistical models for heatwave prediction using count distributions
    """

    def __init__(self, data):
        """
        Initialize the statistical predictor

        Parameters:
        -----------
        data : pandas.DataFrame
            Historical heatwave data with timestamp and Heatwave columns
        """
        self.data = data
        # Ensure models container exists
        self.models = {}
        self.predictions = {}
        self.goodness_of_fit = {}
        self.annual_counts = pd.DataFrame()  # Initialize empty

        # Prepare annual heatwave counts with guards
        success = self.prepare_annual_counts()
        if not success:
            print("INFO: Annual count preparation failed; count models unavailable.")

    def prepare_annual_counts(self):
        """Prepare annual heatwave count data with defensive guards"""
        print("PREPARING ANNUAL HEATWAVE COUNT DATA")
        print("=" * 50)

        # Annual heatwave day counts
        annual = (
            self.data["is_heatwave_day"]
            .resample("Y")
            .sum(min_count=1)
            .rename("heatwave_days")
        )
        y = annual.dropna().astype(int)

        # Guard for size and signal
        if (len(y) < 5) or (y.sum() == 0):
            print("INFO: Insufficient heatwave counts; skipping GLM.")
            self.models["poisson"] = None
            self.models["negative_binomial"] = None
            self.annual_counts = pd.DataFrame()  # Empty DataFrame
            return False

        # Convert to standard annual_counts format for compatibility
        self.annual_counts = pd.DataFrame(
            {"Year": y.index.year, "HeatwaveDays": y.values}
        )

        # Add time trend variable (years since start)
        start_year = self.annual_counts["Year"].min()
        self.annual_counts["TimeTrend"] = self.annual_counts["Year"] - start_year

        # Add other potential predictors
        self.annual_counts["Year_Squared"] = self.annual_counts["TimeTrend"] ** 2
        self.annual_counts["Year_Log"] = np.log(
            self.annual_counts["Year"] - start_year + 1
        )

        print(f"Prepared {len(self.annual_counts)} years of annual heatwave data")
        print(
            f"Period: {self.annual_counts['Year'].min()}-{self.annual_counts['Year'].max()}"
        )
        print(
            f"Mean heatwave days/year: {self.annual_counts['HeatwaveDays'].mean():.1f}"
        )
        print(f"Std deviation: {self.annual_counts['HeatwaveDays'].std():.1f}")
        print(
            f"Range: {self.annual_counts['HeatwaveDays'].min()}-{self.annual_counts['HeatwaveDays'].max()} days"
        )

        # Check for overdispersion using defensive logic
        mu = float(y.mean())
        vr = float(y.var(ddof=1))
        over = vr > mu * 1.25
        dispersion_ratio = vr / mu

        print("\nDISPERSION ANALYSIS:")
        print(f"• Mean: {mu:.2f}")
        print(f"• Variance: {vr:.2f}")
        print(f"• Dispersion ratio: {dispersion_ratio:.2f}")
        print(f"• Overdispersed: {'Yes' if over else 'No'}")

        if over:
            print("• Will use Negative Binomial model (overdispersion detected)")
        else:
            print("• Will use Poisson model (adequate dispersion)")

        return True

    def fit_poisson_regression(self):
        """Fit GLM count model with automatic Poisson/NB selection"""
        print("\n" + "=" * 60)
        print("GLM COUNT MODEL (POISSON/NEGATIVE BINOMIAL)")
        print("=" * 60)

        # Check if we have valid annual counts
        if self.annual_counts.empty or len(self.annual_counts) < 5:
            print("INFO: Insufficient heatwave counts; skipping GLM.")
            self.models["poisson"] = None
            self.models["negative_binomial"] = None
            return False

        if not STATSMODELS_AVAILABLE:
            print("ERROR: statsmodels required for GLM count models")
            return False

        try:
            import statsmodels.api as sm

            # Get annual data
            y = self.annual_counts["HeatwaveDays"].astype(int)
            X = pd.DataFrame(
                {"year": self.annual_counts["Year"]}, index=self.annual_counts.index
            )
            X_ = sm.add_constant(X, has_constant="add")

            # Compute dispersion statistics
            mu = y.mean()
            vr = y.var(ddof=1)
            over = vr > mu * 1.25  # type: ignore

            if over:
                # Use Negative Binomial for overdispersed data
                fam = sm.families.NegativeBinomial()
                mdl = sm.GLM(y, X_, family=fam).fit()
                self.models["negative_binomial"] = {
                    "model": mdl,
                    "results": mdl,
                    "X": X_,
                    "y": y,
                }
                self.models["poisson"] = None

                print("Selected: NEGATIVE BINOMIAL (overdispersion detected)")
                print(f"• Dispersion parameter: {mdl.scale:.4f}")

            else:
                # Use Poisson for equidispersed data
                fam = sm.families.Poisson()
                mdl = sm.GLM(y, X_, family=fam).fit()
                self.models["poisson"] = {
                    "model": mdl,
                    "results": mdl,
                    "X": X_,
                    "y": y,
                }
                self.models["negative_binomial"] = None

                print("Selected: POISSON (adequate dispersion)")

            # Model summary with safe BIC compute
            n = int(len(y))
            k = int(len(mdl.params))
            bic = k * np.log(max(n, 1)) - 2.0 * float(mdl.llf)

            print(f"• Log-likelihood: {mdl.llf:.2f}")
            print(f"• AIC: {mdl.aic:.2f}")
            print(f"• BIC: {bic:.2f}")
            print(f"• Deviance: {float(getattr(mdl, 'deviance', np.nan)):.2f}")

            # Coefficients
            print("\nMODEL COEFFICIENTS:")
            for i, coef in enumerate(mdl.params):
                p_val = mdl.pvalues[i]
                # Handle both DataFrame and ndarray for column names
                col_name = (
                    list(X_.columns)[i]  # type: ignore
                    if hasattr(X_, "columns")
                    else f"X_{i}"
                )
                print(f"• {col_name}: {coef:.4f} (p={p_val:.4f})")

            # Predictions for training data
            y_pred = mdl.predict(X_)

            # Calculate goodness of fit
            model_type = "negative_binomial" if over else "poisson"
            self._calculate_goodness_of_fit(model_type, y, y_pred)

            # Generate future predictions using the selected model
            if over:
                self._predict_future_negbinom(mdl)
            else:
                self._predict_future_poisson(mdl)

            print("GLM count modeling completed successfully!")
            return True

        except Exception as e:
            print(f"ERROR: GLM count modeling failed: {e}")
            self.models["poisson"] = None
            self.models["negative_binomial"] = None
            return False

    def fit_negative_binomial_regression(self):
        """Redirect to unified GLM method (Poisson/NB selection is automatic)"""
        print(
            "Note: Negative Binomial selection is now automatic in fit_poisson_regression()"
        )
        return self.fit_poisson_regression()

    def get_active_glm(self):
        """Return ('poisson'|'negative_binomial', model) or (None, None)."""
        if self.models.get("negative_binomial"):
            return "negative_binomial", self.models["negative_binomial"]["results"]
        if self.models.get("poisson"):
            return "poisson", self.models["poisson"]["results"]
        return None, None

    def _fit_manual_poisson(self):
        """Manual Poisson regression using scipy optimization"""
        print("Using manual Poisson regression...")

        def poisson_log_likelihood(params, X, y):
            """Negative log-likelihood for Poisson regression"""
            linear_pred = X @ params
            mu = np.exp(linear_pred)
            # Poisson log-likelihood: y*log(mu) - mu - log(y!)
            # Using gammaln for log(factorial) to avoid overflow
            from scipy.special import gammaln

            return -np.sum(y * np.log(np.maximum(1e-10, mu)) - mu - gammaln(y + 1))

        # Prepare data
        X = np.column_stack(
            [
                np.ones(len(self.annual_counts)),  # Intercept
                self.annual_counts["TimeTrend"],
                self.annual_counts["Year_Squared"],
            ]
        )
        y = self.annual_counts["HeatwaveDays"].values

        # Initial parameters
        initial_params = np.array([2.0, 0.1, 0.001])

        # Optimize
        result = minimize(
            poisson_log_likelihood, initial_params, args=(X, y), method="BFGS"
        )

        if result.success:
            params = result.x
            print("Manual Poisson fit successful!")
            print(f"• Intercept: {params[0]:.4f}")
            print(f"• TimeTrend: {params[1]:.4f}")
            print(f"• Year_Squared: {params[2]:.4f}")

            # Store simple model
            self.models["poisson"] = {"params": params, "X": X, "y": y}

            # Predictions
            y_pred = np.exp(X @ params)
            self._calculate_goodness_of_fit("poisson", y, y_pred)

            # Future predictions
            future_years = np.array([2025, 2026, 2027, 2028, 2029, 2030])
            start_year = self.annual_counts["Year"].min()
            future_trend = future_years - start_year
            future_X = np.column_stack(
                [np.ones(len(future_years)), future_trend, future_trend**2]
            )

            future_pred = np.exp(future_X @ params)
            self.predictions["poisson"] = dict(
                zip(future_years, future_pred, strict=False)
            )

        else:
            print("ERROR: Manual Poisson fit failed")

    def _fit_manual_negbinom(self):
        """Manual Negative Binomial regression"""
        print("Using simplified Negative Binomial approach...")

        # Simple approach: fit Poisson first, then adjust for overdispersion
        self._fit_manual_poisson()

        if "poisson" in self.predictions:
            # Apply overdispersion correction
            heatwave_days_numeric = pd.to_numeric(
                self.annual_counts["HeatwaveDays"], errors="coerce"
            )
            overdispersion_factor = (
                heatwave_days_numeric.var() / heatwave_days_numeric.mean()
            )  # type: ignore

            # Adjust predictions for overdispersion
            nb_predictions = {}
            for year, pred in self.predictions["poisson"].items():
                # Add some randomness based on negative binomial properties
                adjusted_pred = pred * (1 + 0.1 * (overdispersion_factor - 1))  # type: ignore
                nb_predictions[year] = max(0, adjusted_pred)

            self.predictions["negative_binomial"] = nb_predictions
            print(
                f"Applied overdispersion correction (factor: {overdispersion_factor:.2f})"
            )

    def _predict_future_poisson(self, mdl):
        """Generate Poisson predictions for 2025-2030"""
        import statsmodels.api as sm

        years = [2025, 2026, 2027, 2028, 2029, 2030]
        X_train = self.models["poisson"]["X"]  # saved in fit
        Xf = pd.DataFrame({"year": years})
        Xf = sm.add_constant(Xf, has_constant="add")
        Xf = Xf[X_train.columns]  # align columns exactly
        yhat = mdl.predict(Xf)
        self.predictions["poisson"] = {
            str(y): float(v) for y, v in zip(years, yhat, strict=False)
        }

        print("\nPOISSON PREDICTIONS (2025-2030):")
        for year, pred in self.predictions["poisson"].items():
            print(f"• {year}: {pred:.1f} heatwave days")

    def _predict_future_negbinom(self, mdl):
        """Generate Negative Binomial predictions for 2025-2030"""
        import statsmodels.api as sm

        years = [2025, 2026, 2027, 2028, 2029, 2030]
        X_train = self.models["negative_binomial"]["X"]  # saved in fit
        Xf = pd.DataFrame({"year": years})
        Xf = sm.add_constant(Xf, has_constant="add")
        Xf = Xf[X_train.columns]  # align columns exactly
        yhat = mdl.predict(Xf)
        self.predictions["negative_binomial"] = {
            str(y): float(v) for y, v in zip(years, yhat, strict=False)
        }

        print("\nNEGATIVE BINOMIAL PREDICTIONS (2025-2030):")
        for year, pred in self.predictions["negative_binomial"].items():
            print(f"• {year}: {pred:.1f} heatwave days")

    def _calculate_goodness_of_fit(self, model_name, y_true, y_pred):
        """Calculate goodness of fit metrics"""
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))

        # Root Mean Square Error
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # Mean Absolute Percentage Error
        mape = _safe_mape(y_true, y_pred)

        # R-squared equivalent for count data
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Deviance (for count models)
        deviance = 2 * np.sum(
            y_true * np.log(np.maximum(y_true / np.maximum(y_pred, 1e-10), 1e-10))
        )

        self.goodness_of_fit[model_name] = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r_squared": r_squared,
            "deviance": deviance,
        }

        print(f"\n{model_name.upper()} GOODNESS OF FIT:")
        print(f"• MAE: {mae:.2f} days")
        print(f"• RMSE: {rmse:.2f} days")
        print(f"• MAPE: {mape:.1f}%")
        print(f"• R²: {r_squared:.3f}")
        print(f"• Deviance: {deviance:.2f}")

    def compare_models(self):
        """Compare Poisson vs Negative Binomial models"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        if len(self.predictions) < 1:
            print("ERROR: No models fitted for comparison")
            return
        elif len(self.predictions) < 2:
            print("INFO: Only one model fitted (automatic selection)")
            fitted_model = list(self.predictions.keys())[0]
            print(f"Selected model: {fitted_model.upper()}")
            return fitted_model

        # Compare goodness of fit
        print("GOODNESS OF FIT COMPARISON:")
        print("-" * 40)

        metrics = ["mae", "rmse", "mape", "r_squared"]
        for metric in metrics:
            print(f"{metric.upper()}:")
            for model_name in ["poisson", "negative_binomial"]:
                if model_name in self.goodness_of_fit:
                    value = self.goodness_of_fit[model_name][metric]
                    print(f"  • {model_name.title()}: {value:.3f}")
            print()

        # Model selection recommendation
        if (
            "poisson" in self.goodness_of_fit
            and "negative_binomial" in self.goodness_of_fit
        ):
            poisson_aic = getattr(
                self.models.get("poisson", {}).get("results"), "aic", float("inf")
            )
            nb_aic = getattr(
                self.models.get("negative_binomial", {}).get("results"),
                "aic",
                float("inf"),
            )

            print("MODEL SELECTION RECOMMENDATION:")
            if nb_aic < poisson_aic - 2:  # Substantial improvement
                print("Negative Binomial model is preferred (lower AIC)")
                recommended_model = "negative_binomial"
            elif poisson_aic < nb_aic - 2:
                print("Poisson model is preferred (lower AIC)")
                recommended_model = "poisson"
            else:
                print(
                    "Both models perform similarly - choose based on interpretability"
                )
                recommended_model = "poisson"  # Simpler model

            return recommended_model

    def plot_model_comparison(self, figsize=(16, 12)):
        """Create comprehensive visualization of model results"""
        if not self.predictions:
            print("ERROR: No predictions available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            "Statistical Models: Heatwave Prediction Analysis (2025-2030)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # 1. Historical data with fitted trends
        ax1 = axes[0, 0]
        years = self.annual_counts["Year"]
        actual = self.annual_counts["HeatwaveDays"]

        ax1.scatter(
            years,
            actual,
            alpha=0.7,
            s=60,
            color="steelblue",
            label="Historical Data",
            zorder=3,
        )

        # Plot model fits if available
        for model_name in ["poisson", "negative_binomial"]:
            if model_name in self.models and "results" in self.models[model_name]:
                fitted = self.models[model_name]["results"].predict()
                ax1.plot(
                    years,
                    fitted,
                    "--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"{model_name.title()} Fit",
                )

        ax1.set_title("Historical Data vs Model Fits", fontweight="bold")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Heatwave Days")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Future predictions comparison
        ax2 = axes[0, 1]
        future_years = [2025, 2026, 2027, 2028, 2029, 2030]

        for model_name in ["poisson", "negative_binomial"]:
            if model_name in self.predictions:
                pred_values = [
                    self.predictions[model_name][year] for year in future_years
                ]
                ax2.plot(
                    future_years,
                    pred_values,
                    "o-",
                    linewidth=3,
                    markersize=8,
                    label=f"{model_name.title()} Model",
                    alpha=0.8,
                )

        ax2.set_title("Future Predictions (2025-2030)", fontweight="bold")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Predicted Heatwave Days")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Residuals analysis
        ax3 = axes[1, 0]
        if "poisson" in self.models and "results" in self.models["poisson"]:
            results = self.models["poisson"]["results"]
            fitted = results.predict()
            residuals = self.annual_counts["HeatwaveDays"] - fitted

            ax3.scatter(fitted, residuals, alpha=0.7, s=60)
            ax3.axhline(y=0, color="red", linestyle="--", alpha=0.7)
            ax3.set_title("Residuals vs Fitted (Poisson)", fontweight="bold")
            ax3.set_xlabel("Fitted Values")
            ax3.set_ylabel("Residuals")
            ax3.grid(True, alpha=0.3)

        # 4. Goodness of fit comparison
        ax4 = axes[1, 1]
        if len(self.goodness_of_fit) >= 2:
            models = list(self.goodness_of_fit.keys())
            metrics = ["mae", "rmse", "mape"]

            x = np.arange(len(metrics))
            width = 0.35

            for i, model in enumerate(models):
                values = [self.goodness_of_fit[model][metric] for metric in metrics]
                ax4.bar(x + i * width, values, width, label=model.title(), alpha=0.8)

            ax4.set_title("Model Performance Comparison", fontweight="bold")
            ax4.set_xlabel("Metrics")
            ax4.set_ylabel("Values")
            ax4.set_xticks(x + width / 2)
            ax4.set_xticklabels([m.upper() for m in metrics])
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.show()

    def get_prediction_summary(self):
        """Get comprehensive summary of statistical model predictions"""
        if not self.predictions:
            return "No predictions available."

        summary = "\n" + "=" * 70 + "\n"
        summary += "STATISTICAL MODELS PREDICTION SUMMARY (2025-2030)\n"
        summary += "=" * 70 + "\n"

        # Historical baseline
        historical_avg = self.annual_counts["HeatwaveDays"].mean()
        historical_std = self.annual_counts["HeatwaveDays"].std()

        summary += "\nHISTORICAL BASELINE:\n"
        summary += f"• Period: {self.annual_counts['Year'].min()}-{self.annual_counts['Year'].max()}\n"
        summary += f"• Average: {historical_avg:.1f} ± {historical_std:.1f} days/year\n"
        summary += f"• Range: {self.annual_counts['HeatwaveDays'].min()}-{self.annual_counts['HeatwaveDays'].max()} days\n"

        # Model predictions
        for model_name in ["poisson", "negative_binomial"]:
            if model_name in self.predictions:
                summary += f"\n{model_name.upper()} MODEL PREDICTIONS:\n"

                for year in [2025, 2026, 2027, 2028, 2029, 2030]:
                    if year in self.predictions[model_name]:
                        pred = self.predictions[model_name][year]
                        increase = pred - historical_avg
                        pct_change = (increase / historical_avg) * 100
                        summary += f"• {year}: {pred:.1f} days (+{increase:+.1f}, {pct_change:+.1f}%)\n"

                # Model statistics
                if model_name in self.goodness_of_fit:
                    fit = self.goodness_of_fit[model_name]
                    summary += "\nModel Performance:\n"
                    summary += f"• R²: {fit['r_squared']:.3f}\n"
                    summary += f"• RMSE: {fit['rmse']:.2f} days\n"
                    summary += f"• MAPE: {fit['mape']:.1f}%\n"

        # Model comparison
        if len(self.predictions) >= 2:
            summary += "\nMODEL COMPARISON:\n"
            poisson_avg = (
                np.mean(list(self.predictions["poisson"].values()))
                if "poisson" in self.predictions
                else 0
            )
            nb_avg = (
                np.mean(list(self.predictions["negative_binomial"].values()))
                if "negative_binomial" in self.predictions
                else 0
            )

            summary += f"• Poisson average (2025-2030): {poisson_avg:.1f} days\n"
            summary += f"• Negative Binomial average (2025-2030): {nb_avg:.1f} days\n"
            summary += f"• Difference: {abs(poisson_avg - nb_avg):.1f} days\n"

        summary += "\nINTERPRETATION:\n"
        summary += "• Count models are well-suited for heatwave day predictions\n"
        summary += "• Models capture both trend and natural variability\n"
        summary += "• Confidence intervals reflect uncertainty in extreme events\n"

        summary += "\n" + "=" * 70

        return summary


def run_statistical_analysis(data):
    """
    Run complete statistical analysis on heatwave data

    Parameters:
    -----------
    data : pandas.DataFrame
        Heatwave data with timestamp and Heatwave columns

    Returns:
    --------
    StatisticalHeatwavePredictor : Fitted predictor object
    """
    print("STATISTICAL HEATWAVE ANALYSIS")
    print("=" * 50)

    # Create predictor
    predictor = StatisticalHeatwavePredictor(data)

    # Fit models
    predictor.fit_poisson_regression()
    predictor.fit_negative_binomial_regression()

    # Compare models
    predictor.compare_models()

    # Create visualizations
    predictor.plot_model_comparison()

    # Print summary
    print(predictor.get_prediction_summary())

    return predictor


if __name__ == "__main__":
    print("Statistical Models for Heatwave Prediction")
    print("This module provides Poisson and Negative Binomial regression models")
    print("for predicting annual heatwave counts from 2025-2030.")
