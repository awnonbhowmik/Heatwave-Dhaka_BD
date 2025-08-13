"""
Statistical Analysis Module
"""

from scipy import stats
from scipy.stats import linregress, pearsonr, spearmanr, ttest_ind
from statsmodels.tsa.stattools import adfuller


def comprehensive_statistical_analysis(
    data, combined_data, tree_loss_by_year, annual_temp_stats
):
    """Perform comprehensive statistical analysis"""
    results = {}

    print("=" * 70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 70)

    # 1. Temperature distribution analysis
    temp_col = "Dhaka Temperature [2 m elevation corrected]"
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
    results["stationarity"] = {
        "adf_statistic": adf_result[0],
        "p_value": adf_result[1],
        "critical_values": adf_result[4],
        "is_stationary": adf_result[1] <= 0.05,
    }

    print("\nStationarity Test:")
    print(f"ADF Statistic: {adf_result[0]:.6f}")
    print(f"p-value: {adf_result[1]:.6f}")
    print(f"Stationary: {'Yes' if results['stationarity']['is_stationary'] else 'No'}")

    # 3. Temperature trend analysis
    years = annual_temp_stats["Year"].values
    temps = annual_temp_stats["Dhaka Temperature [2 m elevation corrected]_mean"].values

    slope_temp, intercept_temp, r_temp, p_temp, se_temp = linregress(years, temps)
    results["temperature_trend"] = {
        "slope": slope_temp,
        "r_squared": r_temp**2,
        "p_value": p_temp,
        "total_increase_52years": slope_temp * 52,
        "is_significant": p_temp < 0.05,
    }

    print("\nTemperature Trend (1972-2024):")
    print(f"Rate of change: {slope_temp:.4f} °C/year")
    print(f"Total increase: {slope_temp * 52:.2f} °C over 52 years")
    print(f"R-squared: {r_temp**2:.4f}")
    print(f"Significant: {'Yes' if p_temp < 0.05 else 'No'}")

    # 4. Deforestation-temperature correlation
    valid_years = combined_data[combined_data["umd_tree_cover_loss__ha"] > 0]
    if len(valid_years) > 2:
        temp_vals = valid_years["Dhaka Temperature [2 m elevation corrected]_mean"]
        deforest_vals = valid_years["umd_tree_cover_loss__ha"]

        pearson_corr, pearson_p = pearsonr(temp_vals, deforest_vals)
        spearman_corr, spearman_p = spearmanr(temp_vals, deforest_vals)

        results["deforestation_correlation"] = {
            "pearson": {"correlation": pearson_corr, "p_value": pearson_p},
            "spearman": {"correlation": spearman_corr, "p_value": spearman_p},
        }

        print("\nDeforestation-Temperature Correlation:")
        print(f"Pearson: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"Spearman: {spearman_corr:.4f} (p={spearman_p:.4f})")

    # 5. Heatwave frequency analysis
    annual_heatwave_counts = data[data["Heatwave"]].groupby("Year").size()
    all_years = data["Year"].unique()
    annual_heatwave_counts = annual_heatwave_counts.reindex(
        all_years, fill_value=0
    ).sort_index()

    hw_years = annual_heatwave_counts.index.values
    hw_counts = annual_heatwave_counts.values

    slope_hw, intercept_hw, r_hw, p_hw, se_hw = linregress(hw_years, hw_counts)
    results["heatwave_trend"] = {
        "slope": slope_hw,
        "r_squared": r_hw**2,
        "p_value": p_hw,
        "total_increase_52years": slope_hw * 52,
        "is_significant": p_hw < 0.05,
    }

    print("\nHeatwave Frequency Trend:")
    print(f"Rate of change: {slope_hw:.4f} days/year")
    print(f"Total increase: {slope_hw * 52:.1f} days over 52 years")
    print(f"Significant: {'Yes' if p_hw < 0.05 else 'No'}")

    # 6. Period comparison (pre vs post 2000)
    pre_2000_temp = annual_temp_stats[annual_temp_stats["Year"] < 2000][
        "Dhaka Temperature [2 m elevation corrected]_mean"
    ]
    post_2000_temp = annual_temp_stats[annual_temp_stats["Year"] >= 2000][
        "Dhaka Temperature [2 m elevation corrected]_mean"
    ]

    if len(pre_2000_temp) > 1 and len(post_2000_temp) > 1:
        t_stat, t_p = ttest_ind(post_2000_temp, pre_2000_temp)
        results["period_comparison"] = {
            "pre_2000_mean": pre_2000_temp.mean(),
            "post_2000_mean": post_2000_temp.mean(),
            "difference": post_2000_temp.mean() - pre_2000_temp.mean(),
            "t_test_p": t_p,
            "is_significant": t_p < 0.05,
        }

        print("\nPeriod Comparison (Pre vs Post 2000):")
        print(f"Pre-2000: {pre_2000_temp.mean():.3f} °C")
        print(f"Post-2000: {post_2000_temp.mean():.3f} °C")
        print(f"Difference: {post_2000_temp.mean() - pre_2000_temp.mean():.3f} °C")
        print(f"Significant: {'Yes' if t_p < 0.05 else 'No'}")

    # 7. Heatwave period comparison
    pre_2000_hw = annual_heatwave_counts[annual_heatwave_counts.index < 2000].mean()
    post_2000_hw = annual_heatwave_counts[annual_heatwave_counts.index >= 2000].mean()

    results["heatwave_period_comparison"] = {
        "pre_2000_mean": pre_2000_hw,
        "post_2000_mean": post_2000_hw,
        "increase_days": post_2000_hw - pre_2000_hw,
        "percent_increase": ((post_2000_hw / pre_2000_hw - 1) * 100)
        if pre_2000_hw > 0
        else 0,
    }

    print("\nHeatwave Period Comparison:")
    print(f"Pre-2000: {pre_2000_hw:.1f} days/year")
    print(f"Post-2000: {post_2000_hw:.1f} days/year")
    if pre_2000_hw > 0:
        print(
            f"Increase: {post_2000_hw - pre_2000_hw:.1f} days ({((post_2000_hw / pre_2000_hw - 1) * 100):.1f}%)"
        )

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
