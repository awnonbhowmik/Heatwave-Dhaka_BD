"""
Statistical Analysis Module
Updated to use proper temperature columns and professional output
"""

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from data_dictionary import TEMPERATURE_COLUMNS


def comprehensive_statistical_analysis(
    data, combined_data, tree_loss_by_year, annual_temp_stats
):
    """Perform comprehensive statistical analysis"""
    results = {}

    print("=" * 70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 70)

    # 1. Temperature distribution analysis
    temp_col = TEMPERATURE_COLUMNS["daily_mean"]
    if temp_col not in data.columns:
        raise ValueError(f"Temperature column {temp_col} not found in data")
    temp_series = data[temp_col].dropna()

    results["temperature_distribution"] = {
        "skewness": stats.skew(temp_series),
        "kurtosis": stats.kurtosis(temp_series),
        "normality_test": stats.normaltest(temp_series),
        "is_normal": stats.normaltest(temp_series)[1] > 0.05,
    }

    print("Temperature Distribution Analysis:")
    print(f"Skewness: {results['temperature_distribution']['skewness']:.3f}")
    print(f"Kurtosis: {results['temperature_distribution']['kurtosis']:.3f}")
    print(
        f"Normal Distribution: {'Yes' if results['temperature_distribution']['is_normal'] else 'No'}"
    )

    # 2. Time series stationarity test
    adf_result = adfuller(temp_series, autolag="AIC")
    # Properly unpack adfuller results
    adf_statistic, adf_pvalue, adf_lags, adf_nobs, adf_critical_values = adf_result  # type: ignore

    results["stationarity"] = {
        "adf_statistic": adf_statistic,
        "p_value": adf_pvalue,
        "critical_values": adf_critical_values,
        "is_stationary": adf_pvalue < 0.05,
        "lags_used": adf_lags,
        "nobs_used": adf_nobs,
    }

    print("\nStationarity Test (ADF):")
    print(f"ADF Statistic: {results['stationarity']['adf_statistic']:.3f}")
    print(f"p-value: {results['stationarity']['p_value']:.3f}")
    print(f"Lags used: {results['stationarity']['lags_used']}")
    print(f"Critical values: {results['stationarity']['critical_values']}")
    print(f"Stationary: {'Yes' if results['stationarity']['is_stationary'] else 'No'}")

    # 3. Trend analysis
    time_index = np.arange(len(temp_series))

    # Linear trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        time_index, temp_series
    )

    results["trend_analysis"] = {
        "linear_slope": slope,
        "linear_intercept": intercept,
        "r_squared": r_value**2,  # type: ignore
        "p_value": p_value,
        "std_err": std_err,
        "has_significant_trend": p_value < 0.05,  # type: ignore
        "trend_direction": "warming" if slope > 0 else "cooling",  # type: ignore
        "annual_change": slope * 365.25,  # Convert daily to annual # type: ignore
    }

    print("\nLinear Trend Analysis:")
    print(f"Slope: {results['trend_analysis']['linear_slope']:.6f} °C/day")
    print(f"Annual change: {results['trend_analysis']['annual_change']:.3f} °C/year")
    print(f"R-squared: {results['trend_analysis']['r_squared']:.3f}")
    print(f"p-value: {results['trend_analysis']['p_value']:.3e}")
    print(
        f"Significant trend: {'Yes' if results['trend_analysis']['has_significant_trend'] else 'No'}"
    )

    # 4. Seasonal analysis
    if "timestamp" in data.columns:
        data_with_season = data.copy()
        data_with_season["month"] = data_with_season["timestamp"].dt.month
        monthly_stats = data_with_season.groupby("month")[temp_col].agg(
            ["mean", "std", "count"]
        )

        results["seasonal_analysis"] = {
            "monthly_means": monthly_stats["mean"].to_dict(),
            "monthly_stds": monthly_stats["std"].to_dict(),
            "hottest_month": monthly_stats["mean"].idxmax(),
            "coldest_month": monthly_stats["mean"].idxmin(),
            "seasonal_range": monthly_stats["mean"].max() - monthly_stats["mean"].min(),
        }

        print("\nSeasonal Analysis:")
        print(f"Hottest month: {results['seasonal_analysis']['hottest_month']}")
        print(f"Coldest month: {results['seasonal_analysis']['coldest_month']}")
        print(
            f"Seasonal range: {results['seasonal_analysis']['seasonal_range']:.2f} °C"
        )

    # 5. Extreme values analysis
    q99 = temp_series.quantile(0.99)
    q01 = temp_series.quantile(0.01)
    extreme_hot = temp_series > q99
    extreme_cold = temp_series < q01

    results["extreme_analysis"] = {
        "extreme_hot_threshold": q99,
        "extreme_cold_threshold": q01,
        "extreme_hot_days": int(extreme_hot.sum()),
        "extreme_cold_days": int(extreme_cold.sum()),
        "extreme_hot_percentage": float(extreme_hot.mean() * 100),
        "extreme_cold_percentage": float(extreme_cold.mean() * 100),
    }

    print("\nExtreme Values Analysis:")
    print(
        f"Extreme hot days (>99th percentile): {results['extreme_analysis']['extreme_hot_days']}"
    )
    print(
        f"Extreme cold days (<1st percentile): {results['extreme_analysis']['extreme_cold_days']}"
    )
    print(
        f"Hot threshold: {results['extreme_analysis']['extreme_hot_threshold']:.1f} °C"
    )
    print(
        f"Cold threshold: {results['extreme_analysis']['extreme_cold_threshold']:.1f} °C"
    )

    # 6. Correlation analysis (if tree loss data is available)
    if tree_loss_by_year is not None and not tree_loss_by_year.empty:
        # Match years for correlation
        data_with_year = data.copy()
        data_with_year["year"] = data_with_year["timestamp"].dt.year
        annual_temps = data_with_year.groupby("year")[temp_col].mean()

        common_years = set(annual_temps.index) & set(tree_loss_by_year.index)
        if len(common_years) >= 10:  # Need reasonable sample size
            temp_subset = annual_temps.loc[list(common_years)]
            tree_subset = tree_loss_by_year.loc[list(common_years)]

            correlation = stats.pearsonr(temp_subset, tree_subset.iloc[:, 0])

            results["correlation_analysis"] = {
                "temp_tree_correlation": correlation.statistic,  # type: ignore
                "correlation_p_value": correlation.pvalue,  # type: ignore
                "is_significant": correlation.pvalue < 0.05,  # type: ignore
                "years_analyzed": len(common_years),
            }

            print("\nCorrelation Analysis:")
            print(
                f"Temperature-Deforestation correlation: {results['correlation_analysis']['temp_tree_correlation']:.3f}"
            )
            print(
                f"p-value: {results['correlation_analysis']['correlation_p_value']:.3f}"
            )
            print(
                f"Years analyzed: {results['correlation_analysis']['years_analyzed']}"
            )

    print("\nStatistical analysis completed.")
    print("=" * 70)

    return results


def get_key_insights(results):
    """Extract key statistical insights"""
    insights = []

    # Temperature trend
    temp_trend = results["temperature_trend"]
    insights.append(
        f"Temperature increased by {temp_trend['total_increase_52years']:.2f}°C over 52 years"
    )
    insights.append(f"Warming rate: {temp_trend['slope']:.4f}°C per year")

    # Heatwave trend
    hw_trend = results["heatwave_trend"]
    insights.append(
        f"Heatwave frequency increased by {hw_trend['total_increase_52years']:.1f} days over 52 years"
    )

    # Deforestation correlation
    if "deforestation_correlation" in results:
        deforest_corr = results["deforestation_correlation"]["spearman"]["correlation"]
        insights.append(
            f"Deforestation-temperature correlation: {deforest_corr:.3f} (Spearman)"
        )

    # Period comparison
    if "period_comparison" in results:
        period_diff = results["period_comparison"]["difference"]
        insights.append(
            f"Post-2000 temperature increase: {period_diff:.2f}°C compared to pre-2000"
        )

    return insights
