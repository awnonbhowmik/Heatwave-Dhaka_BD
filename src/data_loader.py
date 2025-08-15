"""
Enhanced Data Loading and Preprocessing Module

Provides comprehensive data loading, validation, preprocessing, and feature engineering
for climate analysis with robust error handling and data quality checks.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global constants
DEFAULT_HEATWAVE_THRESHOLD = 36  # Celsius
TEMP_COLUMN = "Dhaka Temperature [2 m elevation corrected]"
HUMIDITY_COLUMN = "Dhaka Relative Humidity [2 m]"
PRECIP_COLUMN = "Dhaka Precipitation Total"


class DataValidator:
    """Data validation and quality assessment utilities"""

    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file exists and is accessible"""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        if not path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False
        if path.stat().st_size == 0:
            logger.error(f"File is empty: {file_path}")
            return False
        return True

    @staticmethod
    def validate_temperature_data(
        data: pd.DataFrame, temp_col: str = TEMP_COLUMN
    ) -> Dict[str, Any]:
        """Validate temperature data quality and identify issues"""
        validation_results = {"valid": True, "issues": [], "stats": {}, "outliers": []}

        if temp_col not in data.columns:
            validation_results["valid"] = False
            validation_results["issues"].append(
                f"Temperature column '{temp_col}' not found"
            )
            return validation_results

        temp_data = data[temp_col].dropna()

        # Basic statistics
        validation_results["stats"] = {
            "count": len(temp_data),
            "missing": data[temp_col].isna().sum(),
            "mean": temp_data.mean(),
            "std": temp_data.std(),
            "min": temp_data.min(),
            "max": temp_data.max(),
            "range": temp_data.max() - temp_data.min(),
        }

        # Check for reasonable temperature ranges (Dhaka climate)
        if temp_data.min() < -10 or temp_data.max() > 60:
            validation_results["issues"].append(
                "Temperature values outside expected range (-10°C to 60°C)"
            )

        # Identify outliers using IQR method
        Q1 = temp_data.quantile(0.25)
        Q3 = temp_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = temp_data[(temp_data < lower_bound) | (temp_data > upper_bound)]
        validation_results["outliers"] = outliers.tolist()

        # Check for missing values pattern
        missing_pct = (data[temp_col].isna().sum() / len(data)) * 100
        if missing_pct > 5:
            validation_results["issues"].append(
                f"High missing data percentage: {missing_pct:.1f}%"
            )

        logger.info(
            f"Temperature validation: {len(validation_results['issues'])} issues found"
        )
        return validation_results


def handle_missing_values(
    data: pd.DataFrame, method: str = "interpolate"
) -> pd.DataFrame:
    """Handle missing values in the dataset using various methods"""
    logger.info(f"Handling missing values using method: {method}")

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    original_missing = data[numeric_columns].isna().sum().sum()

    if method == "interpolate":
        # Time-aware interpolation for time series data requires DatetimeIndex
        if "timestamp" not in data.columns:
            logger.warning(
                "Timestamp column not found; falling back to linear interpolation"
            )
            data[numeric_columns] = data[numeric_columns].interpolate(
                method="linear", limit_direction="both"
            )
        else:
            # Sort by timestamp to ensure monotonic index
            data = data.sort_values("timestamp").reset_index(drop=True)
            # Temporarily set index to timestamp for time interpolation
            data = data.set_index(pd.to_datetime(data["timestamp"]))
            data[numeric_columns] = data[numeric_columns].interpolate(
                method="time", limit_direction="both"
            )
            data = data.reset_index(drop=True)
    elif method == "knn":
        # KNN imputation for multivariate relationships
        imputer = KNNImputer(n_neighbors=5)
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
    elif method == "drop":
        # Drop rows with any missing values
        data = data.dropna()
    elif method == "forward_fill":
        # Forward fill for time series
        data[numeric_columns] = data[numeric_columns].ffill()

    final_missing = data[numeric_columns].isna().sum().sum()
    logger.info(f"Missing values reduced from {original_missing} to {final_missing}")

    return data


def add_enhanced_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive time-based features"""
    logger.info("Adding enhanced time features...")

    # Basic time features
    data["Year"] = data["timestamp"].dt.year
    data["Month"] = data["timestamp"].dt.month
    data["Day"] = data["timestamp"].dt.day
    data["DayOfYear"] = data["timestamp"].dt.dayofyear
    data["WeekOfYear"] = data["timestamp"].dt.isocalendar().week
    data["DayOfWeek"] = data["timestamp"].dt.dayofweek
    data["Quarter"] = data["timestamp"].dt.quarter

    # Seasonal classification
    data["Season"] = data["timestamp"].dt.month % 12 // 3 + 1
    data["Season_Name"] = data["Season"].map(
        {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
    )

    # Cyclical encoding for better model performance
    data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)
    data["DayOfYear_sin"] = np.sin(2 * np.pi * data["DayOfYear"] / 365)
    data["DayOfYear_cos"] = np.cos(2 * np.pi * data["DayOfYear"] / 365)
    data["DayOfWeek_sin"] = np.sin(2 * np.pi * data["DayOfWeek"] / 7)
    data["DayOfWeek_cos"] = np.cos(2 * np.pi * data["DayOfWeek"] / 7)

    # Time-based flags
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6])
    data["IsMonthStart"] = data["timestamp"].dt.is_month_start
    data["IsMonthEnd"] = data["timestamp"].dt.is_month_end
    data["IsYearStart"] = data["timestamp"].dt.is_year_start
    data["IsYearEnd"] = data["timestamp"].dt.is_year_end

    # Monsoon season (approximate for Bangladesh)
    data["IsMonsoon"] = data["Month"].isin([6, 7, 8, 9])
    data["IsPreMonsoon"] = data["Month"].isin([3, 4, 5])
    data["IsPostMonsoon"] = data["Month"].isin([10, 11])
    data["IsWinter"] = data["Month"].isin([12, 1, 2])

    # Time since epoch (for trend analysis)
    data["Days_Since_Start"] = (data["timestamp"] - data["timestamp"].min()).dt.days
    data["Years_Since_Start"] = data["Days_Since_Start"] / 365.25

    return data


def add_enhanced_temperature_features(
    data: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Add comprehensive temperature-based features"""
    logger.info("Adding enhanced temperature features...")

    temp_col = TEMP_COLUMN

    # Helper to resolve min/max temperature columns robustly
    def _resolve_temp_min_max(
        df: pd.DataFrame, base: str
    ) -> Tuple[Optional[str], Optional[str]]:
        candidates_max = [
            f"{base}.max",
            f"{base}.1",
            f"{base}.2",
            f"{base} Max",
            f"{base} Maximum",
        ]
        candidates_min = [
            f"{base}.min",
            f"{base}.2",
            f"{base}.1",
            f"{base} Min",
            f"{base} Minimum",
        ]

        # If both .1 and .2 exist, try to infer by value ordering on a small sample
        def pick(cols_list):
            for c in cols_list:
                if c in df.columns:
                    return c
            return None

        col1 = f"{base}.1"
        col2 = f"{base}.2"
        inferred_max = inferred_min = None
        if col1 in df.columns and col2 in df.columns:
            sample = df[[col1, col2]].dropna().head(100)
            if not sample.empty:
                # Assume the column with larger mean is max
                if sample[col1].mean() >= sample[col2].mean():
                    inferred_max, inferred_min = col1, col2
                else:
                    inferred_max, inferred_min = col2, col1
        max_col = inferred_max or pick(candidates_max)
        min_col = inferred_min or pick(candidates_min)
        # Guard against picking same column
        if max_col == min_col:
            min_col = None
        return min_col, max_col

    if temp_col in data.columns:
        # Basic heatwave classification
        data["Heatwave"] = data[temp_col] > threshold
        data["Extreme_Heat"] = data[temp_col] > (threshold + 3)  # More severe threshold
        data["Mild_Heat"] = (data[temp_col] > (threshold - 2)) & (
            data[temp_col] <= threshold
        )

        # Temperature categories
        data["Temp_Category"] = pd.cut(
            data[temp_col],
            bins=[-np.inf, 20, 25, 30, 35, threshold, 45, np.inf],
            labels=["Very_Cold", "Cold", "Cool", "Warm", "Hot", "Very_Hot", "Extreme"],
        )

        # Rolling features (if enough data)
        if len(data) > 30:
            data["Temp_MA_7"] = data[temp_col].rolling(window=7, center=True).mean()
            data["Temp_MA_30"] = data[temp_col].rolling(window=30, center=True).mean()
            data["Temp_Std_7"] = data[temp_col].rolling(window=7, center=True).std()
            data["Temp_Std_30"] = data[temp_col].rolling(window=30, center=True).std()

            # Temperature anomalies
            data["Temp_Anomaly_7"] = data[temp_col] - data["Temp_MA_7"]
            data["Temp_Anomaly_30"] = data[temp_col] - data["Temp_MA_30"]

        # Percentile-based features
        data["Temp_Percentile"] = data[temp_col].rank(pct=True)
        data["Is_Top_10_Percent"] = data["Temp_Percentile"] > 0.9
        data["Is_Bottom_10_Percent"] = data["Temp_Percentile"] < 0.1

        # Temperature range calculation (resolve min/max columns)
        min_col, max_col = _resolve_temp_min_max(data, temp_col)
        if max_col and min_col:
            data["Temperature_Range"] = data[max_col] - data[min_col]
            data["Temp_Range_Category"] = pd.cut(
                data["Temperature_Range"],
                bins=[0, 5, 10, 15, np.inf],
                labels=[
                    "Low_Variation",
                    "Moderate_Variation",
                    "High_Variation",
                    "Extreme_Variation",
                ],
            )

    return data


def add_climate_indices(data: pd.DataFrame) -> pd.DataFrame:
    """Add climate indices and composite features"""
    logger.info("Adding climate indices...")

    # Heat Index approximation (if humidity available)
    if HUMIDITY_COLUMN in data.columns and TEMP_COLUMN in data.columns:
        temp_f = data[TEMP_COLUMN] * 9 / 5 + 32  # Convert to Fahrenheit for heat index
        humidity = data[HUMIDITY_COLUMN]

        # Simplified heat index formula
        heat_index_f = (temp_f + humidity) / 2  # Simplified approximation
        data["Heat_Index"] = (heat_index_f - 32) * 5 / 9  # Convert back to Celsius
        data["Heat_Index_Category"] = pd.cut(
            data["Heat_Index"],
            bins=[-np.inf, 27, 32, 41, 54, np.inf],
            labels=[
                "Caution",
                "Extreme_Caution",
                "Danger",
                "Extreme_Danger",
                "Heat_Emergency",
            ],
        )

    # Drought index (if precipitation available)
    if PRECIP_COLUMN in data.columns:
        # 30-day precipitation sum
        data["Precip_30_Sum"] = data[PRECIP_COLUMN].rolling(window=30).sum()
        data["Is_Drought_Period"] = data["Precip_30_Sum"] < data[
            "Precip_30_Sum"
        ].quantile(0.2)

    # Seasonal temperature deviation
    seasonal_means = data.groupby("Season")[TEMP_COLUMN].transform("mean")
    data["Seasonal_Temp_Deviation"] = data[TEMP_COLUMN] - seasonal_means

    return data


def add_quality_flags(data: pd.DataFrame) -> pd.DataFrame:
    """Add data quality flags and indicators"""
    logger.info("Adding data quality flags...")

    # Missing data flags
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data["Has_Missing_Data"] = data[numeric_cols].isna().any(axis=1)
    data["Missing_Count"] = data[numeric_cols].isna().sum(axis=1)

    # Outlier flags (for temperature)
    if TEMP_COLUMN in data.columns:
        temp_data = data[TEMP_COLUMN]
        Q1 = temp_data.quantile(0.25)
        Q3 = temp_data.quantile(0.75)
        IQR = Q3 - Q1
        data["Is_Temp_Outlier"] = (temp_data < (Q1 - 1.5 * IQR)) | (
            temp_data > (Q3 + 1.5 * IQR)
        )

    # Data completeness score
    denom = max(1, len(numeric_cols))
    data["Data_Completeness"] = (1 - data["Missing_Count"] / denom) * 100

    return data


def load_heatwave_data(
    file_path: str = "data/1972_2024_Heatwave_Daily.xlsx",
    validate_data: bool = True,
    heatwave_threshold: float = DEFAULT_HEATWAVE_THRESHOLD,
    handle_missing: str = "interpolate",  # 'interpolate', 'knn', 'drop', 'none'
) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """Enhanced heatwave data loading with validation and preprocessing

    Args:
        file_path: Path to the data file
        validate_data: Whether to perform data validation
        heatwave_threshold: Temperature threshold for heatwave definition
        handle_missing: Method for handling missing values

    Returns:
        Tuple of (processed_data, threshold, validation_results)
    """
    logger.info("📂 Loading heatwave data...")

    # Validate file exists
    if not DataValidator.validate_file_path(file_path):
        raise FileNotFoundError(f"Cannot access file: {file_path}")

    try:
        # Load data with error handling
        if file_path.endswith(".xlsx"):
            data = pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        logger.info(f"Loaded {len(data):,} raw records")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Initialize validation results
    validation_results = {"file_validation": True, "issues": []}

    # Data validation
    if validate_data:
        logger.info("Performing data validation...")
        temp_validation = DataValidator.validate_temperature_data(data)

        validation_results.update({"temperature_validation": temp_validation})

        # Log validation issues
        if temp_validation.get("issues"):
            logger.warning(
                f"Data validation found {len(temp_validation['issues'])} issues"
            )
            for issue in temp_validation["issues"]:
                logger.warning(f"Temperature: {issue}")

    # Ensure timestamp is datetime
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
    else:
        logger.error("No timestamp column found")
        raise ValueError("Timestamp column is required")

    # Handle missing values
    if handle_missing != "none":
        data = handle_missing_values(data, method=handle_missing)

    # Enhanced feature engineering
    data = add_enhanced_time_features(data)
    data = add_enhanced_temperature_features(data, heatwave_threshold)
    data = add_climate_indices(data)
    data = add_quality_flags(data)

    logger.info(
        f"✅ Processed {len(data):,} records ({data['timestamp'].min().year}-{data['timestamp'].max().year})"
    )

    # Count derived features
    derived_features = [
        col
        for col in data.columns
        if any(
            keyword in col.lower()
            for keyword in [
                "_sin",
                "_cos",
                "_ma_",
                "_std_",
                "_anomaly",
                "_percentile",
                "_category",
                "_index",
                "is_",
                "has_",
            ]
        )
    ]
    logger.info(f"Features added: {len(derived_features)} enhanced features")

    return data, heatwave_threshold, validation_results


def load_deforestation_data(
    file_path: str = "data/GFW_Dhaka.csv", validate_data: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Enhanced deforestation data loading with validation

    Args:
        file_path: Path to the deforestation data file
        validate_data: Whether to perform data validation

    Returns:
        Tuple of (raw_data, processed_data, validation_results)
    """
    logger.info("📂 Loading deforestation data...")

    # Validate file exists
    if not DataValidator.validate_file_path(file_path):
        raise FileNotFoundError(f"Cannot access file: {file_path}")

    try:
        deforestation_data = pd.read_csv(file_path)
        logger.info(f"Loaded {len(deforestation_data):,} deforestation records")
    except Exception as e:
        logger.error(f"Failed to load deforestation data: {e}")
        raise

    validation_results = {"file_validation": True, "issues": []}

    # Data validation
    if validate_data:
        logger.info("Validating deforestation data...")

        required_columns = ["Tree_Cover_Loss_Year", "umd_tree_cover_loss__ha"]
        missing_cols = [
            col for col in required_columns if col not in deforestation_data.columns
        ]

        if missing_cols:
            validation_results["issues"].append(
                f"Missing required columns: {missing_cols}"
            )
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for negative values
        if (deforestation_data["umd_tree_cover_loss__ha"] < 0).any():
            validation_results["issues"].append("Found negative tree cover loss values")

        # Check year range
        valid_years = deforestation_data[deforestation_data["Tree_Cover_Loss_Year"] > 0]
        if len(valid_years) == 0:
            validation_results["issues"].append("No valid years found in data")

    # Process tree cover loss by year
    tree_loss_by_year = (
        deforestation_data.groupby("Tree_Cover_Loss_Year")["umd_tree_cover_loss__ha"]
        .agg(["sum", "mean", "std", "count"])
        .reset_index()
    )
    tree_loss_by_year.columns = [
        "Tree_Cover_Loss_Year",
        "Total_Loss_ha",
        "Mean_Loss_ha",
        "Std_Loss_ha",
        "Count_Records",
    ]

    # Filter valid years and add features
    tree_loss_by_year = tree_loss_by_year[
        tree_loss_by_year["Tree_Cover_Loss_Year"] > 0
    ].copy()
    tree_loss_by_year["Year"] = tree_loss_by_year["Tree_Cover_Loss_Year"].astype(int)

    # Add cumulative loss
    tree_loss_by_year = tree_loss_by_year.sort_values("Year")
    tree_loss_by_year["Cumulative_Loss_ha"] = tree_loss_by_year[
        "Total_Loss_ha"
    ].cumsum()

    # Add trend features
    tree_loss_by_year["Loss_Trend_3yr"] = (
        tree_loss_by_year["Total_Loss_ha"].rolling(3).mean()
    )
    tree_loss_by_year["Loss_Trend_5yr"] = (
        tree_loss_by_year["Total_Loss_ha"].rolling(5).mean()
    )

    # Loss rate categories
    if len(tree_loss_by_year) > 0:
        tree_loss_by_year["Loss_Category"] = pd.cut(
            tree_loss_by_year["Total_Loss_ha"],
            bins=[
                0,
                tree_loss_by_year["Total_Loss_ha"].quantile(0.33),
                tree_loss_by_year["Total_Loss_ha"].quantile(0.67),
                np.inf,
            ],
            labels=["Low", "Medium", "High"],
        )

    total_loss = tree_loss_by_year["Total_Loss_ha"].sum()
    logger.info(f"✅ Processed {len(tree_loss_by_year)} years of deforestation data")
    logger.info(
        f"Total forest loss: {total_loss:.0f} hectares ({tree_loss_by_year['Year'].min()}-{tree_loss_by_year['Year'].max()})"
    )

    return deforestation_data, tree_loss_by_year, validation_results


def combine_datasets(
    data: pd.DataFrame, tree_loss_by_year: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Enhanced combination of climate and deforestation datasets with comprehensive statistics

    Args:
        data: Climate data with daily observations
        tree_loss_by_year: Processed deforestation data by year

    Returns:
        Tuple of (combined_data, annual_climate_stats)
    """
    logger.info("Combining climate and deforestation datasets...")

    # Define aggregation functions for different climate variables
    agg_functions = {
        TEMP_COLUMN: [
            "mean",
            "max",
            "min",
            "std",
            "median",
            lambda x: x.quantile(0.95),
        ],
        HUMIDITY_COLUMN: ["mean", "min", "max", "std"],
        PRECIP_COLUMN: ["sum", "mean", "max", "count"],
    }

    # Add optional columns if they exist
    optional_columns = {
        "Dhaka Wind Speed [10 m]": ["mean", "max"],
        "Dhaka Cloud Cover Total": ["mean"],
        "Dhaka Mean Sea Level Pressure [MSL]": ["mean", "std"],
        "Heat_Index": ["mean", "max"],
        "Heatwave": ["sum", "mean"],  # Sum = count of heatwave days, mean = fraction
        "Extreme_Heat": ["sum"],
    }

    # Add existing optional columns to aggregation
    for col, funcs in optional_columns.items():
        if col in data.columns:
            agg_functions[col] = funcs

    # Perform annual aggregation
    annual_climate_stats = data.groupby("Year").agg(agg_functions).round(3)

    # Flatten column names with proper naming
    flattened_cols = []
    for col in annual_climate_stats.columns:
        if isinstance(col, tuple):
            base_name = (
                col[0].replace("[", "_").replace("]", "_").replace(" ", "_").strip("_")
            )
            func_name = col[1] if isinstance(col[1], str) else "p95"
            flattened_cols.append(f"{base_name}_{func_name}")
        else:
            flattened_cols.append(col)

    annual_climate_stats.columns = flattened_cols
    annual_climate_stats = annual_climate_stats.reset_index()

    # Enhanced deforestation column selection
    deforestation_cols = ["Year"]
    available_deforest_cols = [
        col
        for col in ["Total_Loss_ha", "Cumulative_Loss_ha", "Loss_Category"]
        if col in tree_loss_by_year.columns
    ]

    if not available_deforest_cols:
        # Fallback to original column name
        available_deforest_cols = ["umd_tree_cover_loss__ha"]

    deforestation_cols.extend(available_deforest_cols)

    # Merge datasets
    combined_data = pd.merge(
        annual_climate_stats,
        tree_loss_by_year[deforestation_cols],
        on="Year",
        how="left",
    )

    # Fill missing deforestation values (years before 2001)
    for col in available_deforest_cols:
        if col in combined_data.columns:
            # Handle categorical columns differently
            if isinstance(combined_data[col].dtype, pd.CategoricalDtype):
                # For categorical columns, fill with the first category or add 'No Data' category
                if combined_data[col].isna().any():
                    # Add 'No Data' category if not present
                    if "No Data" not in combined_data[col].cat.categories:
                        combined_data[col] = combined_data[col].cat.add_categories(
                            ["No Data"]
                        )
                    combined_data[col] = combined_data[col].fillna("No Data")
            else:
                # For numeric columns, fill with 0
                combined_data[col] = combined_data[col].fillna(0)

    # Add derived features
    combined_data = add_combined_dataset_features(combined_data)

    logger.info(f"✅ Combined datasets: {len(combined_data)} years of data")
    logger.info(
        f"Climate variables: {len([c for c in annual_climate_stats.columns if c != 'Year'])}"
    )
    logger.info(f"Deforestation variables: {len(available_deforest_cols)}")

    return combined_data, annual_climate_stats


def add_combined_dataset_features(combined_data: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the combined dataset"""

    # Temperature trends
    temp_mean_col = f"{TEMP_COLUMN.replace('[', '_').replace(']', '_').replace(' ', '_').strip('_')}_mean"
    if temp_mean_col in combined_data.columns:
        # Multi-year rolling averages
        combined_data["Temp_5yr_Avg"] = (
            combined_data[temp_mean_col].rolling(5, center=True).mean()
        )
        combined_data["Temp_10yr_Avg"] = (
            combined_data[temp_mean_col].rolling(10, center=True).mean()
        )

        # Temperature anomalies from long-term average
        long_term_avg = combined_data[temp_mean_col].mean()
        combined_data["Temp_Anomaly"] = combined_data[temp_mean_col] - long_term_avg

        # Categorize temperature years
        combined_data["Temp_Year_Category"] = pd.cut(
            combined_data[temp_mean_col],
            bins=[
                0,
                combined_data[temp_mean_col].quantile(0.25),
                combined_data[temp_mean_col].quantile(0.75),
                np.inf,
            ],
            labels=["Cool_Year", "Normal_Year", "Hot_Year"],
        )

    # Deforestation impact features
    deforest_cols = [
        col for col in combined_data.columns if "Loss" in col or "loss" in col
    ]
    if deforest_cols:
        main_deforest_col = deforest_cols[
            0
        ]  # Use the first available deforestation column

        # Deforestation categories
        if combined_data[main_deforest_col].max() > 0:
            # Calculate quantiles for non-zero values only
            non_zero_values = combined_data[combined_data[main_deforest_col] > 0][
                main_deforest_col
            ]

            if len(non_zero_values) > 0:
                # Create bins ensuring they are unique
                q33 = non_zero_values.quantile(0.33)
                q67 = non_zero_values.quantile(0.67)

                # Ensure bins are unique by adding small increments if needed
                bins = [0.0]  # Initialize with float type
                if q33 > 0:
                    bins.append(float(q33))
                if q67 > q33:
                    bins.append(float(q67))
                bins.append(float(np.inf))

                # Only create categories if we have more than 2 unique bins
                if len(set(bins[:-1])) > 1:  # Exclude inf for uniqueness check
                    labels = []
                    if len(bins) == 4:
                        labels = [
                            "Low_Deforestation",
                            "Medium_Deforestation",
                            "High_Deforestation",
                        ]
                    elif len(bins) == 3:
                        labels = ["Low_Deforestation", "High_Deforestation"]

                    combined_data["Deforestation_Category"] = pd.cut(
                        combined_data[main_deforest_col],
                        bins=bins,
                        labels=labels,
                        duplicates="drop",  # Handle any remaining duplicates
                    )
                else:
                    # If all non-zero values are the same, create a simple binary category
                    combined_data["Deforestation_Category"] = pd.cut(
                        combined_data[main_deforest_col],
                        bins=[0.0, 0.1, float(np.inf)],
                        labels=["No_Deforestation", "Deforestation"],
                        duplicates="drop",
                    )
            else:
                # If no non-zero values, create a simple category
                combined_data["Deforestation_Category"] = "No_Deforestation"

        # Deforestation rate (year-over-year change)
        combined_data["Deforestation_Rate"] = combined_data[
            main_deforest_col
        ].pct_change()

    # Decade grouping
    combined_data["Decade"] = (combined_data["Year"] // 10) * 10

    # Time period flags
    combined_data["Pre_2000"] = combined_data["Year"] < 2000
    combined_data["Post_2010"] = combined_data["Year"] >= 2010
    combined_data["Recent_Years"] = combined_data["Year"] >= 2015

    return combined_data


def get_dataset_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """Enhanced comprehensive dataset summary with detailed statistics

    Args:
        data: Climate dataset with processed features

    Returns:
        Dictionary with comprehensive summary statistics
    """
    logger.info("Generating comprehensive dataset summary...")

    temp_col = TEMP_COLUMN

    # Basic dataset info
    summary = {
        "dataset_info": {
            "shape": data.shape,
            "total_records": len(data),
            "total_years": data["timestamp"].dt.year.nunique(),
            "time_range": {
                "start": data["timestamp"].min(),
                "end": data["timestamp"].max(),
                "duration_days": (
                    data["timestamp"].max() - data["timestamp"].min()
                ).days,
            },
        }
    }

    # Data quality assessment
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    total_cells = len(data) * len(numeric_columns)
    missing_cells = data[numeric_columns].isna().sum().sum()

    summary["data_quality"] = {
        "total_missing_values": missing_cells,
        "missing_percentage": (missing_cells / total_cells) * 100,
        "complete_rows": len(data.dropna()),
        "completeness_percentage": (len(data.dropna()) / len(data)) * 100,
        "columns_with_missing": data[numeric_columns]
        .isna()
        .sum()[data[numeric_columns].isna().sum() > 0]
        .to_dict(),
    }

    # Temperature analysis
    if temp_col in data.columns:
        temp_data = data[temp_col].dropna()
        summary["temperature_analysis"] = {
            "basic_stats": {
                "count": len(temp_data),
                "mean": temp_data.mean(),
                "median": temp_data.median(),
                "std": temp_data.std(),
                "min": temp_data.min(),
                "max": temp_data.max(),
                "range": temp_data.max() - temp_data.min(),
            },
            "percentiles": {
                "p5": temp_data.quantile(0.05),
                "p25": temp_data.quantile(0.25),
                "p75": temp_data.quantile(0.75),
                "p95": temp_data.quantile(0.95),
                "p99": temp_data.quantile(0.99),
            },
            "distribution": {
                "skewness": temp_data.skew(),
                "kurtosis": temp_data.kurtosis(),
            },
        }

    # Heatwave analysis
    if "Heatwave" in data.columns:
        heatwave_data = data["Heatwave"]
        annual_heatwave_counts = data[data["Heatwave"]].groupby("Year").size()

        summary["heatwave_analysis"] = {
            "total_heatwave_days": heatwave_data.sum(),
            "heatwave_percentage": (heatwave_data.sum() / len(data)) * 100,
            "average_heatwave_days_per_year": annual_heatwave_counts.mean(),
            "max_heatwave_days_in_year": annual_heatwave_counts.max(),
            "years_with_heatwaves": len(
                annual_heatwave_counts[annual_heatwave_counts > 0]
            ),
            "heatwave_trend": {
                "increasing_years": len(
                    annual_heatwave_counts.diff()[annual_heatwave_counts.diff() > 0]
                ),
                "decreasing_years": len(
                    annual_heatwave_counts.diff()[annual_heatwave_counts.diff() < 0]
                ),
            },
        }

    # Seasonal patterns
    if "Season" in data.columns and temp_col in data.columns:
        seasonal_temps = data.groupby("Season")[temp_col].agg(
            ["mean", "std", "min", "max"]
        )
        summary["seasonal_patterns"] = {
            "temperature_by_season": seasonal_temps.to_dict(),
            "hottest_season": seasonal_temps["mean"].idxmax(),
            "coolest_season": seasonal_temps["mean"].idxmin(),
            "seasonal_variation": seasonal_temps["mean"].max()
            - seasonal_temps["mean"].min(),
        }

    # Feature summary
    feature_categories = {
        "time_features": [
            col
            for col in data.columns
            if any(x in col.lower() for x in ["year", "month", "day", "season", "week"])
        ],
        "temperature_features": [col for col in data.columns if "temp" in col.lower()],
        "climate_features": [
            col
            for col in data.columns
            if any(x in col.lower() for x in ["humidity", "precip", "wind", "pressure"])
        ],
        "derived_features": [
            col
            for col in data.columns
            if any(
                x in col.lower()
                for x in ["_sin", "_cos", "_ma", "_std", "_anomaly", "is_", "has_"]
            )
        ],
        "index_features": [col for col in data.columns if "index" in col.lower()],
    }

    summary["feature_analysis"] = {
        "total_features": len(data.columns),
        "feature_categories": {k: len(v) for k, v in feature_categories.items()},
        "numeric_features": len(numeric_columns),
        "categorical_features": len(
            data.select_dtypes(include=["object", "category"]).columns
        ),
    }

    # Correlation insights
    if len(numeric_columns) > 1:
        corr_matrix = data[numeric_columns].corr()

        # Find strong correlations (excluding self-correlations)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if (
                    isinstance(corr_val, (float, int)) and abs(corr_val) > 0.7
                ):  # Strong correlation threshold
                    strong_correlations.append(
                        {
                            "feature_1": corr_matrix.columns[i],
                            "feature_2": corr_matrix.columns[j],
                            "correlation": corr_val,
                        }
                    )

        summary["correlation_analysis"] = {
            "strong_correlations_count": len(strong_correlations),
            "strong_correlations": strong_correlations[
                :10
            ],  # Top 10 to avoid too much data
        }

    logger.info("Dataset summary completed")
    return summary


# Legacy function removed - replaced with enhanced version above
