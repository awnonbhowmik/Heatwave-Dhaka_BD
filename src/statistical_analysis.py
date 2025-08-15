"""
Enhanced Statistical Analysis Module

Provides comprehensive statistical testing, trend analysis, and climate pattern detection
with advanced methodologies for time series analysis and climate research.
"""

import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    f_oneway,
    jarque_bera,
    linregress,
    mannwhitneyu,
    pearsonr,
    shapiro,
    spearmanr,
    ttest_ind,
)
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Statistical constants
SIGNIFICANCE_LEVEL = 0.05
TEMP_COLUMN = "Dhaka Temperature [2 m elevation corrected]"


def enhanced_distribution_analysis(data: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Perform comprehensive distribution analysis on a data column"""
    logger.info(f"Performing distribution analysis on {column}")

    series = data[column].dropna()
    results = {}

    # Basic distribution statistics
    results["basic_stats"] = {
        "count": len(series),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "mode": float(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
        "std": float(series.std()),
        "variance": float(series.var()),  # type: ignore[arg-type]
        "range": float(series.max() - series.min()),
        "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
    }

    # Shape statistics
    results["shape_stats"] = {
        "skewness": float(stats.skew(series)),
        "kurtosis": float(stats.kurtosis(series)),
        "skewness_interpretation": interpret_skewness(stats.skew(series)),
        "kurtosis_interpretation": interpret_kurtosis(stats.kurtosis(series)),
    }

    # Normality tests
    try:
        # D'Agostino and Pearson test
        normaltest_stat, normaltest_p = stats.normaltest(series)

        # Shapiro-Wilk test (for smaller samples)
        if len(series) <= 5000:
            shapiro_stat, shapiro_p = shapiro(series)
        else:
            shapiro_stat, shapiro_p = None, None

        # Jarque-Bera test
        jb_result = jarque_bera(series)
        jb_stat = (
            jb_result[0]
            if isinstance(jb_result, tuple) and len(jb_result) > 0
            else jb_result
        )
        jb_p = (
            jb_result[1] if isinstance(jb_result, tuple) and len(jb_result) > 1 else 1.0
        )

        results["normality_tests"] = {
            "dagostino_pearson": {
                "statistic": float(normaltest_stat),
                "p_value": float(normaltest_p),
            },
            "shapiro_wilk": {
                "statistic": float(shapiro_stat) if shapiro_stat else None,
                "p_value": float(shapiro_p) if shapiro_p else None,
            },
            "jarque_bera": {"statistic": float(jb_stat), "p_value": float(jb_p)},  # type: ignore[arg-type]
            "is_normal_dagostino": normaltest_p > SIGNIFICANCE_LEVEL,
            "is_normal_shapiro": shapiro_p > SIGNIFICANCE_LEVEL if shapiro_p else None,
            "is_normal_jb": jb_p > SIGNIFICANCE_LEVEL,  # type: ignore[operator]
        }
    except Exception as e:
        logger.warning(f"Some normality tests failed: {e}")
        results["normality_tests"] = {"error": str(e)}

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    results["percentiles"] = {
        f"p{p}": float(series.quantile(p / 100)) for p in percentiles
    }

    # Outlier detection
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = series[(series < lower_bound) | (series > upper_bound)]
    results["outliers"] = {
        "count": len(outliers),
        "percentage": (len(outliers) / len(series)) * 100,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "extreme_values": outliers.tolist()[:20],  # Limit to first 20
    }

    return results


def enhanced_stationarity_tests(data: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Perform comprehensive stationarity testing"""
    logger.info(f"Performing stationarity tests on {column}")

    series = data[column].dropna()
    results = {}

    try:
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series, autolag="AIC", regresults=True)
        adf_stat = adf_result[0]
        adf_p = adf_result[1]
        adf_lags = adf_result[2] if len(adf_result) > 2 else 0
        adf_nobs = adf_result[3] if len(adf_result) > 3 else len(series)
        adf_critical = adf_result[4] if len(adf_result) > 4 else {}
        adf_ic = adf_result[5] if len(adf_result) > 5 else 0.0

        results["adf_test"] = {
            "statistic": float(adf_stat),
            "p_value": float(adf_p),
            "lags_used": int(adf_lags),  # type: ignore[arg-type]
            "n_observations": int(adf_nobs),  # type: ignore[arg-type]
            "critical_values": {k: float(v) for k, v in adf_critical.items()},  # type: ignore[attr-defined]
            "ic_best": float(adf_ic),  # type: ignore[arg-type]
            "is_stationary": adf_p <= SIGNIFICANCE_LEVEL,
            "confidence_level": get_adf_confidence_level(adf_stat, adf_critical),  # type: ignore[arg-type]
        }
    except Exception as e:
        logger.warning(f"ADF test failed: {e}")
        results["adf_test"] = {"error": str(e)}

    try:
        # KPSS test
        kpss_stat, kpss_p, kpss_lags, kpss_critical = kpss(series, regression="ct")

        results["kpss_test"] = {
            "statistic": float(kpss_stat),
            "p_value": float(kpss_p),
            "lags_used": int(kpss_lags),
            "critical_values": {k: float(v) for k, v in kpss_critical.items()},
            "is_stationary": kpss_p > SIGNIFICANCE_LEVEL,  # KPSS null: stationary
        }
    except Exception as e:
        logger.warning(f"KPSS test failed: {e}")
        results["kpss_test"] = {"error": str(e)}

    # Overall stationarity conclusion
    adf_stationary = results.get("adf_test", {}).get("is_stationary", False)
    kpss_stationary = results.get("kpss_test", {}).get("is_stationary", False)

    results["stationarity_conclusion"] = {
        "adf_says_stationary": adf_stationary,
        "kpss_says_stationary": kpss_stationary,
        "both_agree": adf_stationary == kpss_stationary,
        "recommendation": get_stationarity_recommendation(
            adf_stationary, kpss_stationary
        ),
    }

    return results


def enhanced_trend_analysis(
    data: pd.DataFrame, x_col: str, y_col: str
) -> Dict[str, Any]:
    """Perform comprehensive trend analysis between two variables"""
    logger.info(f"Performing trend analysis: {x_col} vs {y_col}")

    # Clean data
    clean_data = data[[x_col, y_col]].dropna()
    x = clean_data[x_col].values
    y = clean_data[y_col].values

    results = {}

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    results["linear_regression"] = {
        "slope": float(slope),  # type: ignore[arg-type]
        "intercept": float(intercept),  # type: ignore[arg-type]
        "r_value": float(r_value),  # type: ignore[arg-type]
        "r_squared": float(r_value**2),  # type: ignore[operator,arg-type]
        "p_value": float(p_value),  # type: ignore[arg-type]
        "std_error": float(std_err),  # type: ignore[arg-type]
        "is_significant": p_value < SIGNIFICANCE_LEVEL,  # type: ignore[operator]
        "slope_interpretation": interpret_slope(slope, x_col, y_col),  # type: ignore[arg-type]
        "strength_interpretation": interpret_correlation_strength(abs(r_value)),  # type: ignore[arg-type]
    }

    # Confidence intervals for slope
    n = len(x)
    t_critical = stats.t.ppf(1 - SIGNIFICANCE_LEVEL / 2, n - 2)
    slope_ci_lower = slope - t_critical * std_err
    slope_ci_upper = slope + t_critical * std_err

    results["linear_regression"]["slope_confidence_interval"] = {
        "lower": float(slope_ci_lower),
        "upper": float(slope_ci_upper),
        "confidence_level": (1 - SIGNIFICANCE_LEVEL) * 100,
    }

    # Non-parametric trend test (Mann-Kendall approximation)
    try:
        kendall_result = stats.kendalltau(x, y)
        tau = (
            kendall_result.correlation  # type: ignore[attr-defined]
            if hasattr(kendall_result, "correlation")
            else kendall_result[0]  # type: ignore[index]
        )
        mk_p = kendall_result[1]  # type: ignore[index]
        results["mann_kendall"] = {
            "tau": float(tau),  # type: ignore[arg-type]
            "p_value": float(mk_p),  # type: ignore[arg-type]
            "is_significant": mk_p < SIGNIFICANCE_LEVEL,  # type: ignore[operator]
            "trend_direction": "increasing"
            if tau > 0  # type: ignore[operator]
            else "decreasing"
            if tau < 0  # type: ignore[operator]
            else "no_trend",
        }
    except Exception as e:
        logger.warning(f"Mann-Kendall test failed: {e}")
        results["mann_kendall"] = {"error": str(e)}

    # Time-based projections (if x represents time)
    if "year" in x_col.lower() or "time" in x_col.lower():
        # Note: Future projection functionality can be implemented here
        pass

    return results


def enhanced_correlation_analysis(
    data: pd.DataFrame, columns: List[str], include_categorical: bool = False
) -> Dict[str, Any]:
    """Perform comprehensive correlation analysis"""
    logger.info(f"Performing correlation analysis on {len(columns)} variables")

    results = {}

    # Numeric correlations
    numeric_data = data[columns].select_dtypes(include=[np.number])

    if len(numeric_data.columns) > 1:
        # Pearson correlation
        pearson_corr = numeric_data.corr(method="pearson")

        # Spearman correlation (rank-based)
        spearman_corr = numeric_data.corr(method="spearman")

        # Kendall correlation
        kendall_corr = numeric_data.corr(method="kendall")

        results["correlation_matrices"] = {
            "pearson": pearson_corr.to_dict(),
            "spearman": spearman_corr.to_dict(),
            "kendall": kendall_corr.to_dict(),
        }

        # Find strongest correlations
        strong_correlations = []
        correlation_pairs = []

        for i in range(len(pearson_corr.columns)):
            for j in range(i + 1, len(pearson_corr.columns)):
                var1 = pearson_corr.columns[i]
                var2 = pearson_corr.columns[j]

                pearson_val = pearson_corr.iloc[i, j]
                spearman_val = spearman_corr.iloc[i, j]

                # Statistical significance test
                try:
                    clean_pair_data = data[[var1, var2]].dropna()
                    if len(clean_pair_data) > 3:
                        _, p_pearson = pearsonr(
                            clean_pair_data[var1], clean_pair_data[var2]
                        )
                        _, p_spearman = spearmanr(
                            clean_pair_data[var1], clean_pair_data[var2]
                        )
                    else:
                        p_pearson = p_spearman = 1.0
                except Exception:  # Changed from bare except
                    p_pearson = p_spearman = 1.0

                # Safely convert values to float with proper type handling
                try:
                    pearson_numeric = pd.to_numeric(pearson_val, errors='coerce')
                    if pd.isna(pearson_numeric):
                        pearson_numeric = 0.0
                    else:
                        pearson_numeric = float(pearson_numeric)
                except (TypeError, ValueError, AttributeError):
                    pearson_numeric = 0.0

                try:
                    spearman_numeric = pd.to_numeric(spearman_val, errors='coerce')
                    if pd.isna(spearman_numeric):
                        spearman_numeric = 0.0
                    else:
                        spearman_numeric = float(spearman_numeric)
                except (TypeError, ValueError, AttributeError):
                    spearman_numeric = 0.0

                try:
                    p_pearson_numeric = float(p_pearson)  # type: ignore
                    if pd.isna(p_pearson_numeric):
                        p_pearson_numeric = 1.0
                except (TypeError, ValueError, AttributeError):
                    p_pearson_numeric = 1.0

                try:
                    p_spearman_numeric = float(p_spearman)  # type: ignore
                    if pd.isna(p_spearman_numeric):
                        p_spearman_numeric = 1.0
                except (TypeError, ValueError, AttributeError):
                    p_spearman_numeric = 1.0

                correlation_info = {
                    "variable_1": var1,
                    "variable_2": var2,
                    "pearson_correlation": pearson_numeric,
                    "spearman_correlation": spearman_numeric,
                    "pearson_p_value": p_pearson_numeric,
                    "spearman_p_value": p_spearman_numeric,
                    "pearson_significant": p_pearson_numeric < SIGNIFICANCE_LEVEL,
                    "spearman_significant": p_spearman_numeric < SIGNIFICANCE_LEVEL,
                    "strength_category": interpret_correlation_strength(
                        abs(pearson_numeric)
                    ),
                }

                correlation_pairs.append(correlation_info)

                # Strong correlations (> 0.7)
                if abs(pearson_numeric) > 0.7:
                    strong_correlations.append(correlation_info)

        results["correlation_pairs"] = correlation_pairs
        results["strong_correlations"] = strong_correlations
        results["correlation_summary"] = {
            "total_pairs": len(correlation_pairs),
            "strong_correlations_count": len(strong_correlations),
            "significant_pearson_count": sum(
                1 for p in correlation_pairs if p["pearson_significant"]
            ),
            "significant_spearman_count": sum(
                1 for p in correlation_pairs if p["spearman_significant"]
            ),
        }

    return results


def comprehensive_statistical_analysis(
    data: pd.DataFrame,
    combined_data: pd.DataFrame,
    tree_loss_by_year: pd.DataFrame,
    annual_temp_stats: pd.DataFrame,
) -> Dict[str, Any]:
    """Enhanced comprehensive statistical analysis with advanced methodologies"""
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE STATISTICAL ANALYSIS")
    logger.info("=" * 70)

    results = {}
    temp_col = TEMP_COLUMN

    # 1. Enhanced temperature distribution analysis
    logger.info("1. Temperature Distribution Analysis")
    results["temperature_distribution"] = enhanced_distribution_analysis(data, temp_col)

    # 2. Enhanced stationarity analysis
    logger.info("2. Time Series Stationarity Analysis")
    results["stationarity_analysis"] = enhanced_stationarity_tests(data, temp_col)

    # 3. Enhanced temperature trend analysis
    logger.info("3. Temperature Trend Analysis")
    temp_mean_col = "Dhaka Temperature [2 m elevation corrected]_mean"
    if temp_mean_col in annual_temp_stats.columns:
        results["temperature_trend"] = enhanced_trend_analysis(
            annual_temp_stats, "Year", temp_mean_col
        )

    # 4. Enhanced correlation analysis
    logger.info("4. Climate Variables Correlation Analysis")
    climate_columns = [
        col
        for col in data.columns
        if any(
            keyword in col.lower()
            for keyword in [
                "temperature",
                "humidity",
                "precipitation",
                "wind",
                "pressure",
            ]
        )
    ]

    if len(climate_columns) > 1:
        results["climate_correlations"] = enhanced_correlation_analysis(
            data, climate_columns
        )

    # 5. Deforestation-temperature relationship
    logger.info("5. Deforestation-Temperature Relationship Analysis")
    if (
        "umd_tree_cover_loss__ha" in combined_data.columns
        and temp_mean_col in combined_data.columns
    ):
        valid_deforest_data = combined_data[
            combined_data["umd_tree_cover_loss__ha"] > 0
        ]
        if len(valid_deforest_data) > 3:
            results["deforestation_temperature_analysis"] = enhanced_trend_analysis(
                valid_deforest_data, "umd_tree_cover_loss__ha", temp_mean_col
            )

    # 6. Heatwave frequency analysis
    logger.info("6. Heatwave Frequency Trend Analysis")
    if "Heatwave" in data.columns:
        annual_heatwave_counts = data[data["Heatwave"]].groupby("Year").size()
        all_years = data["Year"].unique()
        annual_heatwave_counts = annual_heatwave_counts.reindex(all_years, fill_value=0)

        heatwave_trend_data = pd.DataFrame(
            {
                "Year": annual_heatwave_counts.index,
                "Heatwave_Days": annual_heatwave_counts.values,
            }
        )

        results["heatwave_trend"] = enhanced_trend_analysis(
            heatwave_trend_data, "Year", "Heatwave_Days"
        )

    # 7. Comparative period analysis
    logger.info("7. Comparative Period Analysis")
    results["period_analysis"] = perform_period_comparisons(
        annual_temp_stats, temp_mean_col
    )

    # 8. Seasonal analysis
    logger.info("8. Seasonal Pattern Analysis")
    if "Season" in data.columns:
        results["seasonal_analysis"] = perform_seasonal_analysis(data, temp_col)

    # 9. Extreme event analysis
    logger.info("9. Extreme Event Analysis")
    results["extreme_events"] = analyze_extreme_events(data, temp_col)

    # 10. Summary insights
    results["summary_insights"] = generate_statistical_insights(results)

    logger.info("=" * 70)
    logger.info("Statistical analysis completed successfully")
    logger.info("=" * 70)

    return results


# Helper functions for interpretation
def interpret_skewness(skewness: float) -> str:
    """Interpret skewness values"""
    if abs(skewness) < 0.5:
        return "approximately_symmetric"
    elif skewness > 0.5:
        return "positively_skewed"
    else:
        return "negatively_skewed"


def interpret_kurtosis(kurtosis: float) -> str:
    """Interpret kurtosis values"""
    if abs(kurtosis) < 0.5:
        return "mesokurtic"  # Normal-like tails
    elif kurtosis > 0.5:
        return "leptokurtic"  # Heavy tails
    else:
        return "platykurtic"  # Light tails


def interpret_correlation_strength(correlation: float) -> str:
    """Interpret correlation strength"""
    abs_corr = abs(correlation)
    if abs_corr < 0.1:
        return "negligible"
    elif abs_corr < 0.3:
        return "weak"
    elif abs_corr < 0.5:
        return "moderate"
    elif abs_corr < 0.7:
        return "strong"
    else:
        return "very_strong"


def interpret_slope(slope: float, x_col: str, y_col: str) -> str:
    """Interpret slope magnitude and direction"""
    if abs(slope) < 0.001:
        return "no_meaningful_trend"
    elif slope > 0:
        return "increasing_trend"
    else:
        return "decreasing_trend"


def get_adf_confidence_level(adf_stat: float, critical_values: Dict[str, float]) -> str:
    """Determine ADF test confidence level"""
    if adf_stat < critical_values.get("1%", float("-inf")):
        return "99%_confidence"
    elif adf_stat < critical_values.get("5%", float("-inf")):
        return "95%_confidence"
    elif adf_stat < critical_values.get("10%", float("-inf")):
        return "90%_confidence"
    else:
        return "insufficient_evidence"


def get_stationarity_recommendation(adf_stationary: bool, kpss_stationary: bool) -> str:
    """Provide stationarity test recommendation"""
    if adf_stationary and kpss_stationary:
        return "series_is_stationary"
    elif not adf_stationary and not kpss_stationary:
        return "series_is_non_stationary"
    elif adf_stationary and not kpss_stationary:
        return "difference_stationary"
    else:
        return "trend_stationary"


def perform_period_comparisons(
    annual_data: pd.DataFrame, temp_col: str
) -> Dict[str, Any]:
    """Perform comprehensive period comparisons"""
    results = {}

    # Define periods
    periods = {
        "pre_2000": annual_data[annual_data["Year"] < 2000],
        "post_2000": annual_data[annual_data["Year"] >= 2000],
        "early_period": annual_data[annual_data["Year"] < 1990],
        "recent_period": annual_data[annual_data["Year"] >= 2010],
        "decade_1970s": annual_data[
            (annual_data["Year"] >= 1970) & (annual_data["Year"] < 1980)
        ],
        "decade_2010s": annual_data[
            (annual_data["Year"] >= 2010) & (annual_data["Year"] < 2020)
        ],
    }

    # Perform comparisons
    for period_name, period_data in periods.items():
        if len(period_data) > 0 and temp_col in period_data.columns:
            results[period_name] = {
                "mean_temperature": float(period_data[temp_col].mean()),
                "std_temperature": float(period_data[temp_col].std()),
                "years_count": len(period_data),
                "min_year": int(period_data["Year"].min()),
                "max_year": int(period_data["Year"].max()),
            }

    # Statistical comparisons
    comparisons = []
    period_names = list(results.keys())

    for i in range(len(period_names)):
        for j in range(i + 1, len(period_names)):
            period1, period2 = period_names[i], period_names[j]
            data1 = periods[period1]
            data2 = periods[period2]

            if len(data1) > 1 and len(data2) > 1:
                try:
                    # T-test
                    t_stat, t_p = ttest_ind(data2[temp_col], data1[temp_col])

                    # Mann-Whitney U test (non-parametric)
                    u_stat, u_p = mannwhitneyu(
                        data2[temp_col], data1[temp_col], alternative="two-sided"
                    )

                    comparison = {
                        "period_1": period1,
                        "period_2": period2,
                        "mean_difference": results[period2]["mean_temperature"]
                        - results[period1]["mean_temperature"],
                        "t_test": {"statistic": float(t_stat), "p_value": float(t_p)},  # type: ignore
                        "mann_whitney": {
                            "statistic": float(u_stat),
                            "p_value": float(u_p),
                        },
                        "significant_parametric": t_p < SIGNIFICANCE_LEVEL,  # type: ignore
                        "significant_nonparametric": u_p < SIGNIFICANCE_LEVEL,  # type: ignore
                    }
                    comparisons.append(comparison)
                except Exception as e:
                    logger.warning(
                        f"Failed comparison between {period1} and {period2}: {e}"
                    )

    results["statistical_comparisons"] = comparisons
    return results


def perform_seasonal_analysis(data: pd.DataFrame, temp_col: str) -> Dict[str, Any]:
    """Perform comprehensive seasonal analysis"""
    results = {}

    # Basic seasonal statistics
    seasonal_stats = data.groupby("Season")[temp_col].agg(
        ["mean", "std", "min", "max", "count"]
    )
    results["seasonal_statistics"] = seasonal_stats.to_dict()

    # ANOVA test for seasonal differences
    try:
        season_groups = [
            data[data["Season"] == season][temp_col].dropna()
            for season in data["Season"].unique()
        ]
        f_stat, f_p = f_oneway(*season_groups)

        results["anova_test"] = {
            "f_statistic": float(f_stat),
            "p_value": float(f_p),
            "significant_seasonal_differences": f_p < SIGNIFICANCE_LEVEL,
        }
    except Exception as e:
        logger.warning(f"ANOVA test failed: {e}")
        results["anova_test"] = {"error": str(e)}

    # Seasonal trend analysis over years
    if "Year" in data.columns:
        seasonal_trends = {}
        for season in data["Season"].unique():
            season_data = data[data["Season"] == season]
            if len(season_data) > 3:
                annual_seasonal = season_data.groupby("Year")[temp_col].mean()
                if len(annual_seasonal) > 3:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        annual_seasonal.index, annual_seasonal.values
                    )
                    seasonal_trends[f"season_{season}"] = {
                        "slope": float(slope),  # type: ignore
                        "r_squared": float(r_value**2),  # type: ignore
                        "p_value": float(p_value),  # type: ignore
                        "is_significant": p_value < SIGNIFICANCE_LEVEL,  # type: ignore
                    }

        results["seasonal_trends"] = seasonal_trends

    return results


def analyze_extreme_events(data: pd.DataFrame, temp_col: str) -> Dict[str, Any]:
    """Analyze extreme temperature events"""
    results = {}

    temp_data = data[temp_col].dropna()

    # Define extreme thresholds
    p95_threshold = temp_data.quantile(0.95)
    p99_threshold = temp_data.quantile(0.99)
    p5_threshold = temp_data.quantile(0.05)
    p1_threshold = temp_data.quantile(0.01)

    # Extreme event counts
    results["extreme_heat_events"] = {
        "p95_threshold": float(p95_threshold),
        "p95_events": int((temp_data > p95_threshold).sum()),
        "p99_threshold": float(p99_threshold),
        "p99_events": int((temp_data > p99_threshold).sum()),
    }

    results["extreme_cold_events"] = {
        "p5_threshold": float(p5_threshold),
        "p5_events": int((temp_data < p5_threshold).sum()),
        "p1_threshold": float(p1_threshold),
        "p1_events": int((temp_data < p1_threshold).sum()),
    }

    # Extreme event trends over time
    if "Year" in data.columns:
        annual_extremes = (
            data.groupby("Year")
            .apply(
                lambda x: pd.Series(
                    {
                        "extreme_heat_days": (x[temp_col] > p95_threshold).sum(),
                        "extreme_cold_days": (x[temp_col] < p5_threshold).sum(),
                        "max_temp": x[temp_col].max(),
                        "min_temp": x[temp_col].min(),
                    }
                )
            )
            .reset_index()
        )

        # Trend analysis for extreme events
        if len(annual_extremes) > 3:
            extreme_trends = {}
            for metric in [
                "extreme_heat_days",
                "extreme_cold_days",
                "max_temp",
                "min_temp",
            ]:
                slope, intercept, r_value, p_value, std_err = linregress(
                    annual_extremes["Year"], annual_extremes[metric]
                )
                extreme_trends[metric] = {
                    "slope": float(slope),  # type: ignore
                    "r_squared": float(r_value**2),  # type: ignore
                    "p_value": float(p_value),  # type: ignore
                    "is_significant": p_value < SIGNIFICANCE_LEVEL,  # type: ignore
                }

            results["extreme_event_trends"] = extreme_trends

    return results


def generate_statistical_insights(results: Dict[str, Any]) -> List[str]:
    """Generate human-readable insights from statistical results"""
    insights = []

    # Temperature distribution insights
    if "temperature_distribution" in results:
        dist_data = results["temperature_distribution"]
        basic_stats = dist_data.get("basic_stats", {})

        insights.append(
            f"Temperature data spans {basic_stats.get('count', 0):,} observations"
        )
        insights.append(f"Mean temperature: {basic_stats.get('mean', 0):.2f}°C")

        if "shape_stats" in dist_data:
            skew_interpretation = dist_data["shape_stats"].get(
                "skewness_interpretation", ""
            )
            insights.append(f"Temperature distribution is {skew_interpretation}")

    # Trend insights
    if "temperature_trend" in results:
        trend_data = results["temperature_trend"]["linear_regression"]
        slope = trend_data.get("slope", 0)
        r_squared = trend_data.get("r_squared", 0)
        is_significant = trend_data.get("is_significant", False)

        if is_significant:
            direction = "increasing" if slope > 0 else "decreasing"
            insights.append(f"Significant {direction} temperature trend detected")
            insights.append(f"Temperature change rate: {slope:.4f}°C per year")
            insights.append(f"Trend explains {r_squared * 100:.1f}% of variance")

    # Stationarity insights
    if "stationarity_analysis" in results:
        stationarity = results["stationarity_analysis"]
        conclusion = stationarity.get("stationarity_conclusion", {})
        recommendation = conclusion.get("recommendation", "")

        insights.append(f"Time series analysis: {recommendation.replace('_', ' ')}")

    # Extreme events insights
    if "extreme_events" in results:
        extreme_data = results["extreme_events"]

        if "extreme_heat_events" in extreme_data:
            heat_events = extreme_data["extreme_heat_events"]
            insights.append(
                f"Extreme heat events (>95th percentile): {heat_events.get('p95_events', 0)} days"
            )

        if "extreme_event_trends" in extreme_data:
            heat_trend = extreme_data["extreme_event_trends"].get(
                "extreme_heat_days", {}
            )
            if heat_trend.get("is_significant", False):
                slope = heat_trend.get("slope", 0)
                direction = "increasing" if slope > 0 else "decreasing"
                insights.append(f"Extreme heat frequency is {direction} over time")

    return insights


def get_key_insights(results):
    """Extract key statistical insights"""
    insights = []
    # Temperature trend summary (if available)
    if "temperature_trend" in results:
        lr = results["temperature_trend"].get("linear_regression", {})
        slope = lr.get("slope")
        r2 = lr.get("r_squared")
        if slope is not None and r2 is not None:
            direction = "increasing" if slope > 0 else "decreasing"
            insights.append(
                f"Temperature trend: {direction} at {slope:.4f}°C/year (R²={r2:.3f})"
            )

    # Heatwave trend (if computed)
    if "heatwave_trend" in results:
        lr = results["heatwave_trend"].get("linear_regression", {})
        slope = lr.get("slope")
        r2 = lr.get("r_squared")
        if slope is not None and r2 is not None:
            direction = "increasing" if slope > 0 else "decreasing"
            insights.append(
                f"Heatwave days trend: {direction} at {slope:.2f} days/year (R²={r2:.3f})"
            )

    # Deforestation-temperature relationship (if computed)
    if "deforestation_temperature_analysis" in results:
        lr = results["deforestation_temperature_analysis"].get("linear_regression", {})
        slope = lr.get("slope")
        r2 = lr.get("r_squared")
        if slope is not None and r2 is not None:
            direction = "positive" if slope > 0 else "negative"
            insights.append(
                f"Deforestation-temperature relationship: {direction} (slope={slope:.4f}, R²={r2:.3f})"
            )

    # Period comparisons summary (if available)
    if "period_analysis" in results:
        pa = results["period_analysis"]
        if "pre_2000" in pa and "post_2000" in pa:
            diff = (
                pa["post_2000"]["mean_temperature"] - pa["pre_2000"]["mean_temperature"]
            )
            insights.append(f"Post-2000 average vs pre-2000: +{diff:.2f}°C")

    # Stationarity insight
    if "stationarity_analysis" in results:
        conc = results["stationarity_analysis"].get("stationarity_conclusion", {})
        rec = conc.get("recommendation")
        if rec:
            insights.append(f"Stationarity conclusion: {rec.replace('_', ' ')}")

    return insights
