"""
Time Series Validation Module
============================

Proper time series validation methods to prevent data leakage and
ensure realistic model evaluation for climate forecasting.

"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TimeSeriesValidator:
    """
    Time series validation with walk-forward analysis and proper temporal splits
    """

    def __init__(self, data: pd.DataFrame, date_column: str = "timestamp"):
        """
        Initialize validator with time series data

        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with datetime index or column
        date_column : str
            Name of the datetime column
        """
        self.data = data.copy()
        self.date_column = date_column

        # Ensure data is sorted by time
        if date_column in data.columns:
            self.data = self.data.sort_values(date_column).reset_index(drop=True)
        else:
            raise ValueError(f"Date column '{date_column}' not found in data")

    def temporal_train_test_split(
        self, test_size: float = 0.2, gap_size: int = 0
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally with optional gap between train and test

        Parameters:
        -----------
        test_size : float
            Proportion of data for testing (default 0.2)
        gap_size : int
            Number of time periods to skip between train and test (default 0)

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame] : Train and test datasets
        """
        n_samples = len(self.data)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test - gap_size

        if n_train <= 0:
            raise ValueError(
                f"Not enough data for train/test split with gap_size={gap_size}"
            )

        train_data = self.data.iloc[:n_train]
        test_data = self.data.iloc[n_train + gap_size :]

        print("Temporal split created:")
        print(
            f"  - Training: {len(train_data)} records ({train_data[self.date_column].min()} to {train_data[self.date_column].max()})"
        )
        if gap_size > 0:
            gap_start = self.data.iloc[n_train][self.date_column]
            gap_end = self.data.iloc[n_train + gap_size - 1][self.date_column]
            print(f"  - Gap: {gap_size} records ({gap_start} to {gap_end})")
        print(
            f"  - Testing: {len(test_data)} records ({test_data[self.date_column].min()} to {test_data[self.date_column].max()})"
        )

        return train_data, test_data

    def walk_forward_validation(
        self,
        model_func,
        target_col: str,
        feature_cols: list[str],
        n_splits: int = 5,
        min_train_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Perform walk-forward validation for time series

        Parameters:
        -----------
        model_func : callable
            Function that takes (X_train, y_train, X_test) and returns predictions
        target_col : str
            Name of target column
        feature_cols : List[str]
            List of feature column names
        n_splits : int
            Number of walk-forward splits
        min_train_size : int
            Minimum training set size (default: 70% of data)

        Returns:
        --------
        Dict[str, Any] : Validation results with metrics for each split
        """
        if min_train_size is None:
            min_train_size = int(len(self.data) * 0.7)

        n_samples = len(self.data)
        test_size = (n_samples - min_train_size) // n_splits

        if test_size <= 0:
            raise ValueError(f"Not enough data for {n_splits} walk-forward splits")

        results = {"split_results": [], "predictions": [], "actuals": [], "dates": []}

        print(f"Walk-forward validation with {n_splits} splits:")

        for i in range(n_splits):
            # Calculate split indices
            train_end = min_train_size + i * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)

            if test_start >= n_samples:
                break

            # Create train/test splits
            train_data = self.data.iloc[:train_end]
            test_data = self.data.iloc[test_start:test_end]

            # Prepare features and targets
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]

            # Make predictions
            try:
                y_pred = model_func(X_train, y_train, X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                split_result = {
                    "split": i + 1,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "train_period": (
                        train_data[self.date_column].min(),
                        train_data[self.date_column].max(),
                    ),
                    "test_period": (
                        test_data[self.date_column].min(),
                        test_data[self.date_column].max(),
                    ),
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                }

                results["split_results"].append(split_result)
                results["predictions"].extend(y_pred.tolist())
                results["actuals"].extend(y_test.tolist())
                results["dates"].extend(test_data[self.date_column].tolist())

                print(f"  Split {i+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

            except Exception as e:
                print(f"  Split {i+1}: Failed with error {e}")
                continue

        # Calculate overall metrics
        if results["predictions"]:
            overall_rmse = np.sqrt(
                mean_squared_error(results["actuals"], results["predictions"])
            )
            overall_mae = mean_absolute_error(
                results["actuals"], results["predictions"]
            )
            overall_r2 = r2_score(results["actuals"], results["predictions"])

            results["overall_metrics"] = {  # type: ignore
                "rmse": overall_rmse,
                "mae": overall_mae,
                "r2": overall_r2,
                "n_predictions": len(results["predictions"]),
            }

            print("\\nOverall Performance:")
            print(f"  - RMSE: {overall_rmse:.4f}")
            print(f"  - MAE: {overall_mae:.4f}")
            print(f"  - R²: {overall_r2:.4f}")
            print(f"  - Total predictions: {len(results['predictions'])}")

        return results

    def expanding_window_validation(
        self,
        model_func,
        target_col: str,
        feature_cols: list[str],
        initial_train_size: int | None = None,
        step_size: int = 12,
    ) -> dict[str, Any]:
        """
        Expanding window validation where training set grows over time

        Parameters:
        -----------
        model_func : callable
            Model function for fitting and prediction
        target_col : str
            Target column name
        feature_cols : List[str]
            Feature column names
        initial_train_size : int
            Initial training set size (default: 60% of data)
        step_size : int
            Step size for expanding the window (default: 12)

        Returns:
        --------
        Dict[str, Any] : Validation results
        """
        if initial_train_size is None:
            initial_train_size = int(len(self.data) * 0.6)

        n_samples = len(self.data)
        results = {"split_results": [], "predictions": [], "actuals": [], "dates": []}

        print(
            f"Expanding window validation starting with {initial_train_size} training samples:"
        )

        split_num = 0
        train_end = initial_train_size

        while train_end + step_size <= n_samples:
            test_start = train_end
            test_end = min(train_end + step_size, n_samples)

            # Create expanding train set and fixed test set
            train_data = self.data.iloc[:train_end]
            test_data = self.data.iloc[test_start:test_end]

            # Prepare features and targets
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]

            try:
                y_pred = model_func(X_train, y_train, X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                split_result = {
                    "split": split_num + 1,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "train_period": (
                        train_data[self.date_column].min(),
                        train_data[self.date_column].max(),
                    ),
                    "test_period": (
                        test_data[self.date_column].min(),
                        test_data[self.date_column].max(),
                    ),
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                }

                results["split_results"].append(split_result)
                results["predictions"].extend(y_pred.tolist())
                results["actuals"].extend(y_test.tolist())
                results["dates"].extend(test_data[self.date_column].tolist())

                print(
                    f"  Window {split_num+1}: Train size={len(train_data)}, RMSE={rmse:.4f}, R²={r2:.4f}"
                )

            except Exception as e:
                print(f"  Window {split_num+1}: Failed with error {e}")

            train_end += step_size
            split_num += 1

        # Calculate overall metrics
        if results["predictions"]:
            overall_rmse = np.sqrt(
                mean_squared_error(results["actuals"], results["predictions"])
            )
            overall_mae = mean_absolute_error(
                results["actuals"], results["predictions"]
            )
            overall_r2 = r2_score(results["actuals"], results["predictions"])

            results["overall_metrics"] = {  # type: ignore
                "rmse": overall_rmse,
                "mae": overall_mae,
                "r2": overall_r2,
                "n_predictions": len(results["predictions"]),
            }

            print("\\nExpanding Window Overall Performance:")
            print(f"  - RMSE: {overall_rmse:.4f}")
            print(f"  - MAE: {overall_mae:.4f}")
            print(f"  - R²: {overall_r2:.4f}")

        return results

    @staticmethod
    def prevent_data_leakage(
        train_data: pd.DataFrame, test_data: pd.DataFrame, feature_operations: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply feature engineering operations separately to train and test sets
        to prevent data leakage

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training dataset
        test_data : pd.DataFrame
            Test dataset
        feature_operations : List[str]
            List of feature engineering operations to apply

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame] : Processed train and test sets
        """
        print("Applying feature operations to prevent data leakage:")

        # Create copies to avoid modifying originals
        train_processed = train_data.copy()
        test_processed = test_data.copy()

        for operation in feature_operations:
            print(f"  - Applying {operation}")

            if operation == "lag_features":
                # Create lagged features within each dataset separately
                for lag in [1, 7, 30]:
                    col_name = f"temp_lag_{lag}"
                    if (
                        "Dhaka Temperature [2 m elevation corrected]"
                        in train_processed.columns
                    ):
                        train_processed[col_name] = train_processed[
                            "Dhaka Temperature [2 m elevation corrected]"
                        ].shift(lag)
                        test_processed[col_name] = test_processed[
                            "Dhaka Temperature [2 m elevation corrected]"
                        ].shift(lag)

            elif operation == "rolling_features":
                # Create rolling features within each dataset separately
                for window in [7, 30]:
                    col_name = f"temp_rolling_{window}"
                    if (
                        "Dhaka Temperature [2 m elevation corrected]"
                        in train_processed.columns
                    ):
                        train_processed[col_name] = (
                            train_processed[
                                "Dhaka Temperature [2 m elevation corrected]"
                            ]
                            .rolling(window)
                            .mean()
                        )
                        test_processed[col_name] = (
                            test_processed[
                                "Dhaka Temperature [2 m elevation corrected]"
                            ]
                            .rolling(window)
                            .mean()
                        )

            elif operation == "normalization":
                # Fit scaler on training data only, apply to both
                from sklearn.preprocessing import StandardScaler

                numeric_cols = train_processed.select_dtypes(
                    include=[np.number]
                ).columns
                scaler = StandardScaler()

                # Fit on training data
                train_scaled = scaler.fit_transform(train_processed[numeric_cols])
                train_processed[numeric_cols] = train_scaled

                # Transform test data using training statistics
                test_scaled = scaler.transform(test_processed[numeric_cols])
                test_processed[numeric_cols] = test_scaled

        return train_processed, test_processed

    def validate_temporal_consistency(self) -> dict[str, Any]:
        """
        Check for temporal consistency in the dataset

        Returns:
        --------
        Dict[str, Any] : Validation results
        """
        results = {"issues": [], "warnings": [], "stats": {}}

        dates = pd.to_datetime(self.data[self.date_column])

        # Check for missing dates
        date_range = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
        missing_dates = set(date_range) - set(dates)

        if missing_dates:
            results["warnings"].append(
                f"Found {len(missing_dates)} missing dates in time series"
            )

        # Check for duplicate dates
        duplicate_dates = dates.duplicated().sum()
        if duplicate_dates > 0:
            results["issues"].append(f"Found {duplicate_dates} duplicate dates")

        # Check temporal ordering
        is_sorted = dates.is_monotonic_increasing
        if not is_sorted:
            results["issues"].append("Data is not sorted chronologically")

        # Calculate time series statistics
        time_diffs = dates.diff().dt.days.dropna()
        results["stats"] = {
            "total_records": len(dates),
            "date_range": (dates.min(), dates.max()),
            "missing_dates": len(missing_dates),
            "duplicate_dates": duplicate_dates,
            "is_chronological": is_sorted,
            "median_time_gap_days": time_diffs.median(),
            "max_time_gap_days": time_diffs.max(),
        }

        return results


def create_simple_model_func(model_type: str = "linear"):
    """
    Create a simple model function for testing validation methods

    Parameters:
    -----------
    model_type : str
        Type of model ('linear', 'rf', 'simple')

    Returns:
    --------
    callable : Model function compatible with validation methods
    """

    def linear_model_func(X_train, y_train, X_test):
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def rf_model_func(X_train, y_train, X_test):
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def simple_model_func(X_train, y_train, X_test):
        # Simple mean model for baseline
        mean_value = y_train.mean()
        return np.full(len(X_test), mean_value)

    if model_type == "linear":
        return linear_model_func
    elif model_type == "rf":
        return rf_model_func
    else:
        return simple_model_func


if __name__ == "__main__":
    print("Time Series Validation Module")
    print("This module provides proper time series validation methods")
    print("to prevent data leakage and ensure realistic model evaluation.")
