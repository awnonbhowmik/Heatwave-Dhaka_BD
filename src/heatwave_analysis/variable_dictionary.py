"""Canonical variable groups used across analyses."""

DESCRIPTIVE_VARIABLES = [
    "tmax", "tmin", "tmean", "day_night_range", "precipitation", "rh_mean",
    "wind_speed_mean", "cloud_cover", "sunshine_duration", "shortwave_radiation",
    "longwave_radiation", "uv_radiation", "pressure_mean", "evapotranspiration",
    "soil_temperature_mean", "soil_moisture_mean",
]

CORRELATION_VARIABLES = DESCRIPTIVE_VARIABLES + ["vpd_mean"]

PRIMARY_ASSOCIATION_PREDICTORS = [
    "rh_mean_lag3_mean", "precipitation_lag7_sum",
    "wind_speed_mean_lag3_mean", "pressure_mean_lag3_mean",
]

TARGET_DERIVED_PREDICTORS = {
    "tmax", "tmin", "tmean", "day_night_range", "heat_index", "apparent_temperature",
    "vpd_mean", "vpd_max", "vpd_min", "cumulative_excess", "maximum_excess",
}
