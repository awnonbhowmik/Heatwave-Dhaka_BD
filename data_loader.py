"""
Data Loading and Preprocessing Module
Updated with proper column definitions and data validation
"""

import warnings

import pandas as pd

from data_dictionary import (
    CLIMATE_COLUMNS,
    DEFORESTATION_COLUMNS,
    HEATWAVE_THRESHOLD,
    TEMPERATURE_COLUMNS,
    validate_deforestation_data,
    validate_temperature_data,
)

warnings.filterwarnings("ignore")


def load_heatwave_data(file_path="data/1972_2024_Heatwave_Daily.xlsx"):
    """Load and preprocess heatwave data with validation"""
    print("Loading heatwave data...")
    data = pd.read_excel(file_path)

    print("Validating temperature data...")
    validation_results = validate_temperature_data(data)

    # Report validation results
    if validation_results["errors"]:
        print("CRITICAL DATA ERRORS:")
        for error in validation_results["errors"]:
            print(f"   • {error}")
        raise ValueError("Critical data validation errors found")

    if validation_results["warnings"]:
        print("WARNING - Data validation warnings:")
        for warning in validation_results["warnings"]:
            print(f"   • {warning}")

    # Print validation statistics
    stats = validation_results["stats"]
    print("Data validation completed:")
    print(f"   - Records: {stats['records_validated']:,}")
    print(
        f"   - Mean temperature: {stats['mean_temperature_avg']:.2f}°C ± {stats['mean_temperature_std']:.2f}"
    )
    print(
        f"   - Daily temperature range: {stats['daily_range_avg']:.2f}°C ± {stats['daily_range_std']:.2f}"
    )

    # Add basic time features
    data["Year"] = data["timestamp"].dt.year
    data["Month"] = data["timestamp"].dt.month
    data["DayOfYear"] = data["timestamp"].dt.dayofyear
    data["Season"] = data["timestamp"].dt.month % 12 // 3 + 1

    # Use correct temperature column for heatwave definition
    temp_col = TEMPERATURE_COLUMNS["daily_mean"]
    data["Heatwave"] = data[temp_col] > HEATWAVE_THRESHOLD

    # Calculate temperature range using min/max columns
    temp_min_col = TEMPERATURE_COLUMNS["daily_min"]
    temp_max_col = TEMPERATURE_COLUMNS["daily_max"]
    data["Temperature_Range"] = data[temp_max_col] - data[temp_min_col]

    print(
        f"Loaded {len(data):,} heatwave records ({data['timestamp'].min().year}-{data['timestamp'].max().year})"
    )
    print(f"   - Using temperature column: {temp_col}")
    print(f"   - Heatwave threshold: {HEATWAVE_THRESHOLD}°C")
    print(
        f"   - Heatwave days found: {data['Heatwave'].sum():,} ({data['Heatwave'].mean() * 100:.1f}% of days)"
    )

    return data, HEATWAVE_THRESHOLD


def load_deforestation_data(file_path="data/GFW_Dhaka.csv"):
    """Load and preprocess deforestation data with validation"""
    print("Loading deforestation data...")
    deforestation_data = pd.read_csv(file_path)

    print("Validating deforestation data...")
    validation_results = validate_deforestation_data(deforestation_data)

    # Report validation results
    if validation_results["errors"]:
        print("CRITICAL DEFORESTATION DATA ERRORS:")
        for error in validation_results["errors"]:
            print(f"   • {error}")
        raise ValueError("Critical deforestation data validation errors found")

    if validation_results["warnings"]:
        print("WARNING - Deforestation data validation warnings:")
        for warning in validation_results["warnings"]:
            print(f"   • {warning}")

    # Use proper column names from data dictionary
    year_col = DEFORESTATION_COLUMNS["year"]
    loss_col = DEFORESTATION_COLUMNS["tree_loss_ha"]

    # Analyze tree cover loss by year
    tree_loss_by_year: pd.DataFrame = (
        deforestation_data.groupby(year_col)[loss_col].sum().reset_index()
    )
    tree_loss_by_year = tree_loss_by_year[tree_loss_by_year[year_col] > 0]
    tree_loss_by_year["Year"] = tree_loss_by_year[year_col].astype(int)
    tree_loss_by_year = tree_loss_by_year.rename(
        columns={loss_col: "umd_tree_cover_loss__ha"}
    )

    # Print validation statistics
    stats = validation_results["stats"]
    print("Deforestation data validation completed:")
    print(f"   - Total records: {stats['total_records']:,}")
    print(f"   - Valid records: {stats['valid_records']:,}")
    print(f"   - Year range: {stats['year_range'][0]}-{stats['year_range'][1]}")
    print(f"   - Total tree loss: {stats['total_tree_loss_ha']:.0f} ha")
    print(f"   - Data completeness: {stats['data_completeness'] * 100:.1f}%")
    print(
        f"   - Missing years: {len(stats['missing_years'])} years (1972-2000, 2024-2030)"
    )

    print(
        f"Loaded {len(tree_loss_by_year)} years of deforestation data (Total: {tree_loss_by_year['umd_tree_cover_loss__ha'].sum():.0f} ha)"
    )
    return deforestation_data, tree_loss_by_year


def combine_datasets(data, tree_loss_by_year):
    """Combine climate and deforestation data with proper column handling"""
    print("Combining datasets...")

    # Use proper temperature column from data dictionary
    temp_col = TEMPERATURE_COLUMNS["daily_mean"]
    humidity_col = CLIMATE_COLUMNS["humidity"]
    precip_col = CLIMATE_COLUMNS["precipitation"]

    # Prepare annual temperature data using correct columns
    annual_temp_stats = (
        data.groupby("Year")
        .agg(
            {
                temp_col: ["mean", "max", "min", "std"],
                humidity_col: "mean",
                precip_col: "sum",
            }
        )
        .round(3)
    )

    # Flatten column names
    annual_temp_stats.columns = [
        "_".join(col).strip() for col in annual_temp_stats.columns
    ]
    annual_temp_stats = annual_temp_stats.reset_index()

    # Merge with deforestation data
    combined_data = pd.merge(
        annual_temp_stats,
        tree_loss_by_year[["Year", "umd_tree_cover_loss__ha"]],
        on="Year",
        how="left",
    )

    # Handle missing deforestation values with warning
    missing_deforest = combined_data["umd_tree_cover_loss__ha"].isnull().sum()
    if missing_deforest > 0:
        print(
            f"WARNING - Filling {missing_deforest} missing deforestation values with 0 (years before 2001)"
        )
    combined_data["umd_tree_cover_loss__ha"] = combined_data[
        "umd_tree_cover_loss__ha"
    ].fillna(0)

    print("Datasets combined successfully!")
    print(f"   - Combined data shape: {combined_data.shape}")
    print(
        f"   - Years with deforestation data: {(combined_data['umd_tree_cover_loss__ha'] > 0).sum()}"
    )
    print(
        f"   - Years without deforestation data: {(combined_data['umd_tree_cover_loss__ha'] == 0).sum()}"
    )

    return combined_data, annual_temp_stats


def get_dataset_summary(data):
    """Get comprehensive dataset summary statistics using proper columns"""
    temp_col = TEMPERATURE_COLUMNS["daily_mean"]

    # Validate that required columns exist
    if temp_col not in data.columns:
        raise ValueError(f"Temperature column {temp_col} not found in data")

    summary = {
        "shape": data.shape,
        "time_range": (data["timestamp"].min(), data["timestamp"].max()),
        "total_days": len(data),
        "total_years": data["timestamp"].dt.year.nunique(),
        "missing_values": data.isnull().sum().sum(),
        "heatwave_days": data["Heatwave"].sum() if "Heatwave" in data.columns else 0,
        "heatwave_percentage": (
            data["Heatwave"].mean() * 100 if "Heatwave" in data.columns else 0
        ),
        "temperature_stats": {
            "mean": data[temp_col].mean(),
            "median": data[temp_col].median(),
            "std": data[temp_col].std(),
            "min": data[temp_col].min(),
            "max": data[temp_col].max(),
            "p95": data[temp_col].quantile(0.95),
            "p99": data[temp_col].quantile(0.99),
        },
        "temperature_column_used": temp_col,
        "heatwave_threshold": HEATWAVE_THRESHOLD,
    }
    return summary


def print_data_summary(summary):
    """Print formatted dataset summary"""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(
        f"Time Range: {summary['time_range'][0].strftime('%Y-%m-%d')} to {summary['time_range'][1].strftime('%Y-%m-%d')}"
    )
    print(f"Total Days: {summary['total_days']:,}")
    print(f"Total Years: {summary['total_years']}")
    print(f"Missing Values: {summary['missing_values']:,}")
    print(f"\nTEMPERATURE STATISTICS ({summary['temperature_column_used']}):")
    temp_stats = summary["temperature_stats"]
    print(f"   - Mean: {temp_stats['mean']:.2f}°C")
    print(f"   - Median: {temp_stats['median']:.2f}°C")
    print(f"   - Std Dev: {temp_stats['std']:.2f}°C")
    print(f"   - Range: {temp_stats['min']:.2f}°C to {temp_stats['max']:.2f}°C")
    print(f"   - 95th percentile: {temp_stats['p95']:.2f}°C")
    print(f"   - 99th percentile: {temp_stats['p99']:.2f}°C")
    print("\nHEATWAVE STATISTICS:")
    print(f"   - Threshold: {summary['heatwave_threshold']}°C")
    print(f"   - Heatwave Days: {summary['heatwave_days']:,}")
    print(f"   - Heatwave Percentage: {summary['heatwave_percentage']:.2f}%")
    print("=" * 60)
