"""
Statistical Analysis Module
Updated to use proper temperature columns and professional output
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from data_dictionary import TEMPERATURE_COLUMNS

# Statistical analysis constants
ALPHA_LEVEL = 0.05  # Significance level for statistical tests
NORMALITY_ALPHA = 0.05  # Alpha level for normality tests
STATIONARITY_ALPHA = 0.05  # Alpha level for stationarity tests
EXTREME_PERCENTILES = {"hot": 0.99, "cold": 0.01}
DAYS_PER_YEAR = 365.25  # Including leap years
MIN_CORRELATION_SAMPLE_SIZE = 10
DECIMAL_PRECISION = 3


def describe_array(x: np.ndarray) -> dict:
    """Summarize a 1-D numeric array safely using scipy.stats.describe."""
    # Drop NaNs for stable stats
    x = x[np.isfinite(x)]
    # Handle empty arrays
    if x.size == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "var": None,
            "skew": None,
            "kurt": None,
        }
    # Use attribute access to avoid unpack bugs
    d = stats.describe(x, nan_policy="omit")
    # Build a dict result
    return {
        "n": int(d.nobs),
        "min": float(d.minmax[0]),
        "max": float(d.minmax[1]),
        "mean": float(d.mean),
        "var": float(d.variance),
        "skew": float(d.skewness),
        "kurt": float(d.kurtosis),
    }


def monthly_aggregates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Build monthly aggregates from the chosen canonical column."""
    # Compute mean, max, min by month start
    m = df[[col]].resample("MS").agg(["mean", "max", "min"])
    # Flatten MultiIndex into predictable names
    m.columns = [f"{col}_{c}" for c in m.columns]
    # Return the flattened frame
    return m


def _validate_temperature_data(data: pd.DataFrame) -> pd.Series:
    """Validate and extract temperature data for analysis.

    Args:
        data: DataFrame containing temperature data

    Returns:
        Series with validated temperature data

    Raises:
        ValueError: If temperature column is not found
    """
    temp_col = TEMPERATURE_COLUMNS["daily_mean"]
    if temp_col not in data.columns:
        raise ValueError(f"Temperature column {temp_col} not found in data")

    temp_series = data[temp_col].dropna()
    if len(temp_series) == 0:
        raise ValueError("No valid temperature data found")

    return temp_series


def _analyze_temperature_distribution(temp_series: pd.Series) -> dict[str, Any]:
    """Analyze temperature distribution characteristics.

    Args:
        temp_series: Temperature data series

    Returns:
        Dictionary with distribution analysis results
    """
    normality_test = stats.normaltest(temp_series)

    results = {
        "skewness": float(stats.skew(temp_series)),
        "kurtosis": float(stats.kurtosis(temp_series)),
        "normality_test": {
            "statistic": float(normality_test.statistic),
            "p_value": float(normality_test.pvalue),
        },
        "is_normal": float(normality_test.pvalue) > NORMALITY_ALPHA,
    }

    print("Temperature Distribution Analysis:")
    print(f"Skewness: {results['skewness']:.{DECIMAL_PRECISION}f}")
    print(f"Kurtosis: {results['kurtosis']:.{DECIMAL_PRECISION}f}")
    print(f"Normal Distribution: {'Yes' if results['is_normal'] else 'No'}")

    return results


def _analyze_stationarity(temp_series: pd.Series) -> dict[str, Any]:
    """Analyze time series stationarity using ADF test.

    Args:
        temp_series: Temperature data series

    Returns:
        Dictionary with stationarity analysis results
    """
    adf_result = adfuller(temp_series, autolag="AIC")
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1] 
    adf_lags = adf_result[2]
    adf_nobs = adf_result[3]
    adf_critical_values = adf_result[4] if len(adf_result) > 4 else {}

    results = {
        "adf_statistic": float(adf_statistic),
        "p_value": float(adf_pvalue),
        "critical_values": (
            {level: float(value) for level, value in adf_critical_values.items()}
            if hasattr(adf_critical_values, 'items')
            else {}
        ),
        "is_stationary": float(adf_pvalue) < STATIONARITY_ALPHA,
        "lags_used": int(adf_lags),  # type: ignore
        "nobs_used": int(adf_nobs),  # type: ignore
    }

    print("\nStationarity Test (ADF):")
    print(f"ADF Statistic: {results['adf_statistic']:.{DECIMAL_PRECISION}f}")
    print(f"p-value: {results['p_value']:.{DECIMAL_PRECISION}f}")
    print(f"Lags used: {results['lags_used']}")
    print(f"Critical values: {results['critical_values']}")
    print(f"Stationary: {'Yes' if results['is_stationary'] else 'No'}")

    return results


def _analyze_trend(temp_series: pd.Series) -> dict[str, Any]:
    """Analyze linear trend in temperature data.

    Args:
        temp_series: Temperature data series

    Returns:
        Dictionary with trend analysis results
    """
    time_index = np.arange(len(temp_series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        time_index, temp_series
    )

    results = {
        "linear_slope": float(slope),  # pyright: ignore[reportArgumentType]
        "linear_intercept": float(intercept),  # pyright: ignore[reportArgumentType]
        "r_squared": float(r_value**2),  # pyright: ignore[reportOperatorIssue]
        "p_value": float(p_value),  # pyright: ignore[reportArgumentType]
        "std_err": float(std_err),  # pyright: ignore[reportArgumentType]
        "has_significant_trend": float(p_value)  # pyright: ignore[reportArgumentType]
        < ALPHA_LEVEL,  # pyright: ignore[reportArgumentType]
        "trend_direction": (
            "warming"
            if slope > 0  # pyright: ignore[reportOperatorIssue]
            else "cooling"  # pyright: ignore[reportOperatorIssue]
        ),
        "annual_change": float(
            slope * DAYS_PER_YEAR  # pyright: ignore[reportOperatorIssue]
        ),  # pyright: ignore[reportOperatorIssue]
    }

    print("\nLinear Trend Analysis:")
    print(f"Slope: {results['linear_slope']:.6f} °C/day")
    print(f"Annual change: {results['annual_change']:.{DECIMAL_PRECISION}f} °C/year")
    print(f"R-squared: {results['r_squared']:.{DECIMAL_PRECISION}f}")
    print(f"p-value: {results['p_value']:.3e}")
    print(f"Significant trend: {'Yes' if results['has_significant_trend'] else 'No'}")

    return results


def _analyze_seasonal_patterns(
    data: pd.DataFrame, temp_col: str
) -> dict[str, Any] | None:
    """Analyze seasonal patterns in temperature data.

    Args:
        data: DataFrame with temperature data and timestamp
        temp_col: Name of temperature column

    Returns:
        Dictionary with seasonal analysis results or None if timestamp missing
    """
    if "timestamp" not in data.columns:
        return None

    data_with_season = data.copy()
    data_with_season["month"] = data_with_season["timestamp"].dt.month
    monthly_stats = data_with_season.groupby("month")[temp_col].agg(
        ["mean", "std", "count"]
    )

    results = {
        "monthly_means": monthly_stats["mean"].to_dict(),
        "monthly_stds": monthly_stats["std"].to_dict(),
        "hottest_month": int(monthly_stats["mean"].idxmax()),
        "coldest_month": int(monthly_stats["mean"].idxmin()),
        "seasonal_range": float(
            monthly_stats["mean"].max() - monthly_stats["mean"].min()
        ),
    }

    print("\nSeasonal Analysis:")
    print(f"Hottest month: {results['hottest_month']}")
    print(f"Coldest month: {results['coldest_month']}")
    print(f"Seasonal range: {results['seasonal_range']:.2f} °C")

    return results


def _analyze_extreme_values(temp_series: pd.Series) -> dict[str, Any]:
    """Analyze extreme temperature values.

    Args:
        temp_series: Temperature data series

    Returns:
        Dictionary with extreme values analysis results
    """
    q_hot = temp_series.quantile(EXTREME_PERCENTILES["hot"])
    q_cold = temp_series.quantile(EXTREME_PERCENTILES["cold"])
    extreme_hot = temp_series > q_hot
    extreme_cold = temp_series < q_cold

    results = {
        "extreme_hot_threshold": float(q_hot),
        "extreme_cold_threshold": float(q_cold),
        "extreme_hot_days": int(extreme_hot.sum()),
        "extreme_cold_days": int(extreme_cold.sum()),
        "extreme_hot_percentage": float(extreme_hot.mean() * 100),
        "extreme_cold_percentage": float(extreme_cold.mean() * 100),
    }

    print("\nExtreme Values Analysis:")
    print(
        f"Extreme hot days (>{EXTREME_PERCENTILES['hot']*100:.0f}th percentile): {results['extreme_hot_days']}"
    )
    print(
        f"Extreme cold days (<{EXTREME_PERCENTILES['cold']*100:.0f}st percentile): {results['extreme_cold_days']}"
    )
    print(f"Hot threshold: {results['extreme_hot_threshold']:.1f} °C")
    print(f"Cold threshold: {results['extreme_cold_threshold']:.1f} °C")

    return results


def _analyze_correlation_with_deforestation(
    data: pd.DataFrame, tree_loss_by_year: pd.DataFrame | None, temp_col: str
) -> dict[str, Any] | None:
    """Analyze correlation between temperature and deforestation.

    Args:
        data: DataFrame with temperature data
        tree_loss_by_year: DataFrame with deforestation data
        temp_col: Name of temperature column

    Returns:
        Dictionary with correlation analysis results or None if insufficient data
    """
    if tree_loss_by_year is None or tree_loss_by_year.empty:
        return None

    # Match years for correlation
    data_with_year = data.copy()
    data_with_year["year"] = data_with_year["timestamp"].dt.year
    annual_temps = data_with_year.groupby("year")[temp_col].mean()

    common_years = set(annual_temps.index) & set(tree_loss_by_year.index)
    if len(common_years) < MIN_CORRELATION_SAMPLE_SIZE:
        print(
            f"\nInsufficient data for correlation analysis: {len(common_years)} common years"
        )
        return None

    temp_subset = annual_temps.loc[list(common_years)]
    tree_subset = tree_loss_by_year.loc[list(common_years)]

    correlation = stats.pearsonr(temp_subset, tree_subset.iloc[:, 0])

    results = {
        "temp_tree_correlation": float(
            correlation.statistic  # pyright: ignore[reportAttributeAccessIssue]
        ),
        "correlation_p_value": float(
            correlation.pvalue  # pyright: ignore[reportAttributeAccessIssue]
        ),
        "is_significant": float(
            correlation.pvalue  # pyright: ignore[reportAttributeAccessIssue]
        )
        < ALPHA_LEVEL,
        "years_analyzed": len(common_years),
    }

    print("\nCorrelation Analysis:")
    print(
        f"Temperature-Deforestation correlation: {results['temp_tree_correlation']:.{DECIMAL_PRECISION}f}"
    )
    print(f"p-value: {results['correlation_p_value']:.{DECIMAL_PRECISION}f}")
    print(f"Years analyzed: {results['years_analyzed']}")

    return results


def comprehensive_statistical_analysis(
    data: pd.DataFrame,
    combined_data: pd.DataFrame,
    tree_loss_by_year: pd.DataFrame | None,
    annual_temp_stats: pd.DataFrame,
) -> dict[str, Any]:
    """Perform comprehensive statistical analysis using modular approach.

    Args:
        data: DataFrame with daily climate data
        combined_data: DataFrame with combined climate and deforestation data
        tree_loss_by_year: DataFrame with annual deforestation data
        annual_temp_stats: DataFrame with annual temperature statistics

    Returns:
        Dictionary containing all statistical analysis results

    Raises:
        ValueError: If temperature data is invalid or missing
    """
    print("=" * 70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 70)

    # Validate and extract temperature data
    temp_series = _validate_temperature_data(data)
    temp_col = TEMPERATURE_COLUMNS["daily_mean"]

    # Run individual analysis components using helper functions
    results = {}

    # 1. Temperature distribution analysis
    results["temperature_distribution"] = _analyze_temperature_distribution(temp_series)

    # 2. Time series stationarity test
    results["stationarity"] = _analyze_stationarity(temp_series)

    # 3. Trend analysis
    results["trend_analysis"] = _analyze_trend(temp_series)

    # 4. Seasonal analysis (optional - depends on timestamp column)
    seasonal_results = _analyze_seasonal_patterns(data, temp_col)
    if seasonal_results:
        results["seasonal_analysis"] = seasonal_results

    # 5. Extreme values analysis
    results["extreme_analysis"] = _analyze_extreme_values(temp_series)

    # 6. Correlation analysis (optional - depends on deforestation data)
    correlation_results = _analyze_correlation_with_deforestation(
        data, tree_loss_by_year, temp_col
    )
    if correlation_results:
        results["correlation_analysis"] = correlation_results

    print("\nStatistical analysis completed.")
    print("=" * 70)

    return results


def get_key_insights(results: dict[str, Any]) -> list[str]:
    """Extract key statistical insights from analysis results.

    Args:
        results: Dictionary containing statistical analysis results

    Returns:
        List of key insights as formatted strings
    """
    insights = []

    # Temperature trend insights
    if "trend_analysis" in results:
        trend = results["trend_analysis"]
        insights.append(
            f"Temperature warming trend: {trend['annual_change']:.{DECIMAL_PRECISION}f}°C per year"
        )
        insights.append(
            f"Trend significance: {'Significant' if trend['has_significant_trend'] else 'Not significant'} "
            f"(p-value: {trend['p_value']:.3e})"
        )

        # Calculate total change over typical study period (52 years mentioned in insights)
        total_change = trend["annual_change"] * 52
        insights.append(
            f"Estimated temperature increase over 52 years: {total_change:.2f}°C"
        )

    # Distribution insights
    if "temperature_distribution" in results:
        dist = results["temperature_distribution"]
        insights.append(
            f"Temperature distribution: {'Normal' if dist['is_normal'] else 'Non-normal'} "
            f"(skewness: {dist['skewness']:.{DECIMAL_PRECISION}f})"
        )

    # Extreme values insights
    if "extreme_analysis" in results:
        extremes = results["extreme_analysis"]
        insights.append(
            f"Extreme hot days (>{EXTREME_PERCENTILES['hot']*100:.0f}th percentile): "
            f"{extremes['extreme_hot_days']} days ({extremes['extreme_hot_percentage']:.1f}%)"
        )

    # Seasonal insights
    if "seasonal_analysis" in results:
        seasonal = results["seasonal_analysis"]
        insights.append(
            f"Seasonal temperature range: {seasonal['seasonal_range']:.1f}°C "
            f"(hottest: month {seasonal['hottest_month']}, coldest: month {seasonal['coldest_month']})"
        )

    # Stationarity insights
    if "stationarity" in results:
        stationarity = results["stationarity"]
        insights.append(
            f"Time series stationarity: {'Stationary' if stationarity['is_stationary'] else 'Non-stationary'} "
            f"(ADF p-value: {stationarity['p_value']:.{DECIMAL_PRECISION}f})"
        )

    # Correlation insights
    if "correlation_analysis" in results:
        corr = results["correlation_analysis"]
        significance = "significant" if corr["is_significant"] else "not significant"
        insights.append(
            f"Temperature-deforestation correlation: {corr['temp_tree_correlation']:.{DECIMAL_PRECISION}f} "
            f"({significance}, {corr['years_analyzed']} years analyzed)"
        )

    return insights
