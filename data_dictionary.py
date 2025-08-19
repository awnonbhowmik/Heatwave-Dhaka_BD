"""
Data Dictionary for Heatwave Analysis Project
============================================

This module defines data structures, column mappings, and validation rules
for the Dhaka heatwave analysis project.

"""

# Temperature Column Definitions
TEMPERATURE_COLUMNS = {
    "daily_mean": "Dhaka Temperature [2 m elevation corrected]",  # Range: 16.3-40.2°C (Daily average)
    "daily_min": "Dhaka Temperature [2 m elevation corrected].1",  # Range: 6.5-29.5°C (Daily minimum)
    "daily_max": "Dhaka Temperature [2 m elevation corrected].2",  # Range: 10.6-34.0°C (Daily maximum)
}

# Climate Variable Definitions
CLIMATE_COLUMNS = {
    "temperature_mean": TEMPERATURE_COLUMNS["daily_mean"],
    "temperature_min": TEMPERATURE_COLUMNS["daily_min"],
    "temperature_max": TEMPERATURE_COLUMNS["daily_max"],
    "precipitation": "Dhaka Precipitation Total",
    "humidity": "Dhaka Relative Humidity [2 m]",
    "wind_speed": "Dhaka Wind Speed [10 m]",
    "cloud_cover": "Dhaka Cloud Cover Total",
    "pressure": "Dhaka Mean Sea Level Pressure [MSL]",
    "sunshine": "Dhaka Sunshine Duration",
    "shortwave_radiation": "Dhaka Shortwave Radiation",
    "longwave_radiation": "Dhaka Longwave Radiation",
    "uv_radiation": "Dhaka UV Radiation",
    "evapotranspiration": "Dhaka Evapotranspiration",
    "vapor_pressure_deficit": "Dhaka Vapor Pressure Deficit [2 m]",
}

# Deforestation Column Definitions
DEFORESTATION_COLUMNS = {
    "year": "Tree_Cover_Loss_Year",
    "tree_loss_ha": "umd_tree_cover_loss__ha",
    "co2_emissions": "gfw_gross_emissions_co2e_all_gases__Mg_x",
    "net_flux": "gfw_net_flux_co2e__Mg",
}

# Data Availability Periods
DATA_PERIODS = {
    "climate_data": {
        "start_year": 1972,
        "end_year": 2024,
        "total_years": 53,
        "source": "1972_2024_Heatwave_Daily.xlsx",
    },
    "deforestation_data": {
        "start_year": 2001,
        "end_year": 2023,
        "total_years": 23,
        "source": "GFW_Dhaka.csv",
        "gap_years": list(range(1972, 2001)) + list(range(2024, 2031)),  # Missing years
    },
    "prediction_period": {"start_year": 2025, "end_year": 2030, "total_years": 6},
}

# Constants and Thresholds
HEATWAVE_THRESHOLD = 36.0  # °C - Temperature threshold for heatwave definition
MIN_VALID_TEMPERATURE = -10.0  # °C - Minimum realistic temperature for Dhaka
MAX_VALID_TEMPERATURE = 50.0  # °C - Maximum realistic temperature for Dhaka

# Data Quality Thresholds
DATA_QUALITY = {
    "max_missing_ratio": 0.05,  # Maximum 5% missing values allowed
    "temperature_daily_range_max": 25.0,  # Max daily temperature range (max-min)
    "temperature_daily_range_min": 2.0,  # Min daily temperature range
    "humidity_range": (0, 100),  # Valid humidity percentage range
    "precipitation_max": 500.0,  # Maximum daily precipitation (mm)
}


def validate_temperature_data(df):
    """
    Validate temperature data quality and consistency

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing temperature data

    Returns:
    --------
    dict : Validation results with warnings and errors
    """
    validation_results = {"errors": [], "warnings": [], "stats": {}}

    temp_mean_col = TEMPERATURE_COLUMNS["daily_mean"]
    temp_min_col = TEMPERATURE_COLUMNS["daily_min"]
    temp_max_col = TEMPERATURE_COLUMNS["daily_max"]

    # Check if columns exist
    required_cols = [temp_mean_col, temp_min_col, temp_max_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation_results["errors"].append(
            f"Missing temperature columns: {missing_cols}"
        )
        return validation_results

    # Get temperature data
    temp_mean = df[temp_mean_col]
    temp_min = df[temp_min_col]
    temp_max = df[temp_max_col]

    # Check for reasonable temperature ranges
    if temp_mean.min() < MIN_VALID_TEMPERATURE:
        validation_results["warnings"].append(
            f"Mean temperature below minimum: {temp_mean.min():.2f}°C"
        )
    if temp_mean.max() > MAX_VALID_TEMPERATURE:
        validation_results["warnings"].append(
            f"Mean temperature above maximum: {temp_mean.max():.2f}°C"
        )

    # Check temperature relationships (min <= mean <= max)
    invalid_min_mean = (temp_min > temp_mean).sum()
    invalid_mean_max = (temp_mean > temp_max).sum()

    if invalid_min_mean > 0:
        validation_results["errors"].append(
            f"Daily min > mean in {invalid_min_mean} records"
        )
    if invalid_mean_max > 0:
        validation_results["errors"].append(
            f"Daily mean > max in {invalid_mean_max} records"
        )

    # Check daily temperature ranges
    daily_range = temp_max - temp_min
    invalid_range = (
        (daily_range < DATA_QUALITY["temperature_daily_range_min"])
        | (daily_range > DATA_QUALITY["temperature_daily_range_max"])
    ).sum()

    if invalid_range > 0:
        validation_results["warnings"].append(
            f"Unusual daily temperature ranges in {invalid_range} records"
        )

    # Calculate statistics
    validation_results["stats"] = {
        "mean_temperature_avg": temp_mean.mean(),
        "mean_temperature_std": temp_mean.std(),
        "daily_range_avg": daily_range.mean(),
        "daily_range_std": daily_range.std(),
        "records_validated": len(df),
        "missing_temp_mean": temp_mean.isnull().sum(),
        "missing_temp_min": temp_min.isnull().sum(),
        "missing_temp_max": temp_max.isnull().sum(),
    }

    return validation_results


def validate_deforestation_data(df):
    """
    Validate deforestation data quality and temporal coverage

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing deforestation data

    Returns:
    --------
    dict : Validation results
    """
    validation_results = {"errors": [], "warnings": [], "stats": {}}

    year_col = DEFORESTATION_COLUMNS["year"]
    loss_col = DEFORESTATION_COLUMNS["tree_loss_ha"]

    # Check if columns exist
    if year_col not in df.columns:
        validation_results["errors"].append(f"Missing year column: {year_col}")
        return validation_results
    if loss_col not in df.columns:
        validation_results["errors"].append(f"Missing tree loss column: {loss_col}")
        return validation_results

    # Filter valid records (non-null years)
    valid_df = df[df[year_col].notna() & (df[year_col] > 0)]

    if len(valid_df) == 0:
        validation_results["errors"].append("No valid deforestation records found")
        return validation_results

    # Check temporal coverage
    year_range = valid_df[year_col]
    min_year = int(year_range.min())
    max_year = int(year_range.max())

    expected_start = DATA_PERIODS["deforestation_data"]["start_year"]
    expected_end = DATA_PERIODS["deforestation_data"]["end_year"]

    if min_year != expected_start:
        validation_results["warnings"].append(
            f"Deforestation data starts {min_year}, expected {expected_start}"
        )
    if max_year != expected_end:
        validation_results["warnings"].append(
            f"Deforestation data ends {max_year}, expected {expected_end}"
        )

    # Check for negative tree loss values
    negative_loss = (valid_df[loss_col] < 0).sum()
    if negative_loss > 0:
        validation_results["errors"].append(
            f"Found {negative_loss} negative tree loss values"
        )

    # Calculate statistics
    validation_results["stats"] = {
        "total_records": len(df),
        "valid_records": len(valid_df),
        "year_range": (min_year, max_year),
        "total_tree_loss_ha": valid_df[loss_col].sum(),
        "avg_annual_loss_ha": valid_df[loss_col].mean(),
        "missing_years": DATA_PERIODS["deforestation_data"]["gap_years"],
        "data_completeness": len(valid_df) / (max_year - min_year + 1),
    }

    return validation_results


def get_feature_columns():
    """
    Get standardized feature column definitions for modeling

    Returns:
    --------
    dict : Feature column mappings by category
    """
    return {
        "target": TEMPERATURE_COLUMNS["daily_mean"],
        "climate_features": [
            CLIMATE_COLUMNS["humidity"],
            CLIMATE_COLUMNS["precipitation"],
            CLIMATE_COLUMNS["wind_speed"],
            CLIMATE_COLUMNS["cloud_cover"],
            CLIMATE_COLUMNS["pressure"],
        ],
        "deforestation_features": [
            DEFORESTATION_COLUMNS["tree_loss_ha"],
        ],
        "temporal_features": [
            "Year",
            "Month",
            "DayOfYear",
            "Season",
            "Month_sin",
            "Month_cos",
            "DayOfYear_sin",
            "DayOfYear_cos",
        ],
        "derived_features": [
            "Heat_Index",
            "Temperature_Range",
            "Temp_lag_1",
            "Temp_lag_7",
            "Temp_lag_30",
            "Temp_rolling_7",
            "Temp_rolling_30",
            "Cumulative_Deforestation",
        ],
    }


def create_data_summary():
    """Create a comprehensive data summary"""
    summary = f"""
    HEATWAVE ANALYSIS DATA DICTIONARY
    ================================

    DATASET OVERVIEW:
    • Climate Data: {DATA_PERIODS['climate_data']['total_years']} years ({DATA_PERIODS['climate_data']['start_year']}-{DATA_PERIODS['climate_data']['end_year']})
    • Deforestation Data: {DATA_PERIODS['deforestation_data']['total_years']} years ({DATA_PERIODS['deforestation_data']['start_year']}-{DATA_PERIODS['deforestation_data']['end_year']})
    • Prediction Period: {DATA_PERIODS['prediction_period']['total_years']} years ({DATA_PERIODS['prediction_period']['start_year']}-{DATA_PERIODS['prediction_period']['end_year']})

    TEMPERATURE COLUMNS:
    • Mean Temperature: {TEMPERATURE_COLUMNS['daily_mean']} (Primary target variable)
    • Minimum Temperature: {TEMPERATURE_COLUMNS['daily_min']} (Daily minimum)
    • Maximum Temperature: {TEMPERATURE_COLUMNS['daily_max']} (Daily maximum)

    HEATWAVE DEFINITION:
    • Threshold: {HEATWAVE_THRESHOLD}°C (Daily mean temperature)
    • Based on: World Meteorological Organization guidelines for Bangladesh

    DATA LIMITATIONS:
    • Deforestation data missing: 1972-2000 (29 years) and 2024-2030 (7 years)
    • Climate projections require extrapolation beyond available data
    • Urban heat island effects not explicitly modeled

    VALIDATION RULES:
    • Temperature range: {MIN_VALID_TEMPERATURE}°C to {MAX_VALID_TEMPERATURE}°C
    • Missing data threshold: <{DATA_QUALITY['max_missing_ratio']*100}%
    • Daily temperature range: {DATA_QUALITY['temperature_daily_range_min']}-{DATA_QUALITY['temperature_daily_range_max']}°C
    """
    return summary


if __name__ == "__main__":
    print(create_data_summary())
