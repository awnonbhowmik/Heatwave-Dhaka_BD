"""
Improved LSTM Module
===================

Enhanced LSTM implementation with proper sequence lengths for climate data,
regularization, and uncertainty quantification.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from data_dictionary import TEMPERATURE_COLUMNS
from time_series_validation import TimeSeriesValidator
from uncertainty_quantification import UncertaintyQuantifier

try:
    # Modern Keras API imports (TF 2.16+)
    from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.api.layers import LSTM, BatchNormalization, Dense, Dropout
    from keras.api.models import Sequential
    from keras.api.optimizers import Adam
except ImportError:
    # Fallback to tensorflow.keras for older versions
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

TENSORFLOW_AVAILABLE = True


class ImprovedLSTMPredictor:
    """
    Enhanced LSTM predictor with proper sequence lengths and climate-specific features
    """

    def __init__(self, data, tree_loss_by_year=None):
        """
        Initialize LSTM predictor with validated data

        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with climate measurements
        tree_loss_by_year : pd.DataFrame, optional
            Deforestation data
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM modeling")

        self.data = data
        self.tree_loss_by_year = tree_loss_by_year
        self.models = {}
        self.forecasts = {}
        self.scalers = {}

        # Use proper temperature column
        self.temp_col = TEMPERATURE_COLUMNS["daily_mean"]
        if self.temp_col not in data.columns:
            raise ValueError(f"Temperature column {self.temp_col} not found in data")

        # Climate-appropriate sequence length (1+ years for seasonal patterns)
        self.sequence_length = 365  # 1 year of daily data for climate patterns

        print(f"Initialized LSTM predictor with {len(data):,} records")
        print(f"Using temperature column: {self.temp_col}")
        print(f"Sequence length: {self.sequence_length} days (climate-appropriate)")

    def prepare_features(self):
        """
        Prepare features for LSTM modeling with climate-specific engineering
        """
        print("Preparing climate features for LSTM...")

        feature_data = self.data.copy()

        # Temporal features (essential for climate)
        feature_data["Year"] = feature_data["timestamp"].dt.year
        feature_data["Month"] = feature_data["timestamp"].dt.month
        feature_data["DayOfYear"] = feature_data["timestamp"].dt.dayofyear

        # Cyclical encoding for seasonal patterns
        feature_data["Month_sin"] = np.sin(2 * np.pi * feature_data["Month"] / 12)
        feature_data["Month_cos"] = np.cos(2 * np.pi * feature_data["Month"] / 12)
        feature_data["DayOfYear_sin"] = np.sin(
            2 * np.pi * feature_data["DayOfYear"] / 365
        )
        feature_data["DayOfYear_cos"] = np.cos(
            2 * np.pi * feature_data["DayOfYear"] / 365
        )

        # Climate variables (avoid target leakage by not using lagged temperature)
        climate_features = [
            "Dhaka Relative Humidity [2 m]",
            "Dhaka Precipitation Total",
            "Dhaka Wind Speed [10 m]",
            "Dhaka Cloud Cover Total",
            "Dhaka Mean Sea Level Pressure [MSL]",
            "Month_sin",
            "Month_cos",
            "DayOfYear_sin",
            "DayOfYear_cos",
        ]

        # Filter available features
        available_features = [
            col for col in climate_features if col in feature_data.columns
        ]

        print(f"Available climate features: {len(available_features)}")
        for i, feat in enumerate(available_features):
            print(f"  {i+1}. {feat}")

        # Add deforestation data if available (without leakage)
        if self.tree_loss_by_year is not None:
            # Add annual deforestation data
            annual_deforest = self.tree_loss_by_year.set_index("Year")[
                "umd_tree_cover_loss__ha"
            ]
            feature_data["Annual_Deforestation"] = (
                feature_data["Year"].map(annual_deforest).fillna(0)
            )
            available_features.append("Annual_Deforestation")

        self.feature_data = feature_data
        self.available_features = available_features

        print(f"Feature preparation completed. Shape: {feature_data.shape}")
        return feature_data

    def create_sequences(self, features, target, sequence_length):
        """
        Create sequences for LSTM training with proper time series structure

        Parameters:
        -----------
        features : np.ndarray
            Feature array
        target : np.ndarray
            Target array
        sequence_length : int
            Length of input sequences

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : X sequences and y targets
        """
        X, y = [], []

        for i in range(sequence_length, len(features)):
            # Use previous sequence_length days to predict current day
            X.append(features[i - sequence_length : i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def fit_improved_lstm(self, validation_split=0.2, epochs=50):
        """
        Fit improved LSTM model with proper architecture and validation

        Parameters:
        -----------
        validation_split : float
            Proportion of data for validation
        epochs : int
            Maximum training epochs

        Returns:
        --------
        dict : Model results and predictions
        """
        print("=" * 70)
        print("IMPROVED LSTM CLIMATE FORECASTING")
        print("=" * 70)

        if not hasattr(self, "feature_data"):
            self.prepare_features()

        # Prepare data
        target_col = self.temp_col
        feature_cols = self.available_features

        # Create feature matrix and target vector
        X_data = self.feature_data[feature_cols].values
        y_data = self.feature_data[target_col].values

        print(f"Input data shape: {X_data.shape}")
        print(f"Target data shape: {y_data.shape}")

        # Handle missing values
        X_data = np.nan_to_num(X_data, nan=0.0)
        y_data = np.nan_to_num(y_data, nan=np.nanmean(y_data))

        # Scale features and target separately
        self.scalers["features"] = MinMaxScaler(feature_range=(0, 1))
        self.scalers["target"] = MinMaxScaler(feature_range=(0, 1))

        X_scaled = self.scalers["features"].fit_transform(X_data)
        y_scaled = self.scalers["target"].fit_transform(y_data.reshape(-1, 1)).flatten()

        print("Data scaling completed")

        # Create sequences
        X_sequences, y_sequences = self.create_sequences(
            X_scaled, y_scaled, self.sequence_length
        )

        print(f"Sequences created: X={X_sequences.shape}, y={y_sequences.shape}")

        if len(X_sequences) < 100:
            print(
                "WARNING: Very few sequences available. Consider reducing sequence_length."
            )

        # Time series split (avoid random splitting)
        TimeSeriesValidator(
            pd.DataFrame(X_sequences.reshape(len(X_sequences), -1))
        )

        split_idx = int(len(X_sequences) * (1 - validation_split))
        X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_val = y_sequences[:split_idx], y_sequences[split_idx:]

        print(f"Training set: {len(X_train)} sequences")
        print(f"Validation set: {len(X_val)} sequences")

        # Build improved LSTM architecture
        print("Building LSTM architecture...")

        # Clear any previous models
        try:
            from keras.api import backend

            backend.clear_session()
        except ImportError:
            tf.keras.backend.clear_session()
        tf.random.set_seed(42)

        model = Sequential(
            [
                # Input layer
                LSTM(
                    64,
                    return_sequences=True,
                    input_shape=(self.sequence_length, len(feature_cols)),
                    dropout=0.2,
                    recurrent_dropout=0.2,
                ),
                BatchNormalization(),
                # Second LSTM layer
                LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                BatchNormalization(),
                # Third LSTM layer (no return sequences)
                LSTM(16, dropout=0.2, recurrent_dropout=0.2),
                BatchNormalization(),
                # Dense layers
                Dense(32, activation="relu"),
                Dropout(0.3),
                Dense(16, activation="relu"),
                Dropout(0.2),
                Dense(1),  # Single output for temperature prediction
            ]
        )

        # Compile with appropriate loss and optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])

        print("Model compiled successfully")
        print(f"Total parameters: {model.count_params():,}")

        # Advanced callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=8, min_lr=1e-7, verbose=1
        )

        callbacks = [early_stopping, reduce_lr]

        # Train the model
        print("Training LSTM model...")

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False,  # Maintain temporal order
        )

        print("LSTM training completed!")

        # Evaluate model
        print("Evaluating model performance...")

        train_pred = model.predict(X_train, verbose=0)
        val_pred = model.predict(X_val, verbose=0)

        # Inverse transform predictions
        train_pred_orig = self.scalers["target"].inverse_transform(train_pred).flatten()
        val_pred_orig = self.scalers["target"].inverse_transform(val_pred).flatten()

        y_train_orig = (
            self.scalers["target"].inverse_transform(y_train.reshape(-1, 1)).flatten()
        )
        y_val_orig = (
            self.scalers["target"].inverse_transform(y_val.reshape(-1, 1)).flatten()
        )

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred_orig))
        val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
        train_mae = mean_absolute_error(y_train_orig, train_pred_orig)
        val_mae = mean_absolute_error(y_val_orig, val_pred_orig)
        train_r2 = r2_score(y_train_orig, train_pred_orig)
        val_r2 = r2_score(y_val_orig, val_pred_orig)

        print("Training Metrics:")
        print(f"  - RMSE: {train_rmse:.4f}°C")
        print(f"  - MAE: {train_mae:.4f}°C")
        print(f"  - R²: {train_r2:.4f}")

        print("Validation Metrics:")
        print(f"  - RMSE: {val_rmse:.4f}°C")
        print(f"  - MAE: {val_mae:.4f}°C")
        print(f"  - R²: {val_r2:.4f}")

        # Generate future predictions
        print("Generating future climate predictions...")

        # Use last sequence for forecasting
        last_sequence = X_scaled[-self.sequence_length :].reshape(
            1, self.sequence_length, len(feature_cols)
        )

        # Generate predictions for next 6 years (2190 days)
        future_days = 365 * 6  # 6 years
        future_predictions = []

        current_sequence = last_sequence.copy()

        for _day in range(future_days):
            # Predict next day
            next_pred = model.predict(current_sequence, verbose=0)
            future_predictions.append(next_pred[0, 0])

            # Update sequence for next prediction (simplified approach)
            # In practice, would need to estimate future feature values
            next_features = current_sequence[0, -1, :].copy()
            next_features[0] = next_pred[0, 0]  # Update temperature prediction

            # Shift sequence window
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = next_features

        # Convert predictions back to original scale
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions_orig = (
            self.scalers["target"].inverse_transform(future_predictions).flatten()
        )

        # Calculate annual averages
        annual_forecasts = {}
        for year in range(6):
            year_start = year * 365
            year_end = (year + 1) * 365
            if year_end <= len(future_predictions_orig):
                annual_avg = future_predictions_orig[year_start:year_end].mean()
                annual_forecasts[str(2025 + year)] = float(annual_avg)

        print("Annual Temperature Forecasts (LSTM):")
        historical_avg = self.feature_data[target_col].mean()
        for year, temp in annual_forecasts.items():
            increase = temp - historical_avg
            print(f"  {year}: {temp:.2f}°C ({increase:+.2f}°C from historical)")

        # Add uncertainty quantification
        uncertainty_quantifier = UncertaintyQuantifier()

        # Bootstrap confidence intervals for future predictions
        residuals = y_val_orig - val_pred_orig
        prediction_uncertainty = (
            uncertainty_quantifier.time_series_prediction_intervals(
                residuals, future_predictions_orig, len(future_predictions_orig)
            )
        )

        # Store results
        lstm_results = {
            "model": model,
            "scalers": self.scalers,
            "sequence_length": self.sequence_length,
            "features_used": feature_cols,
            "training_history": history.history,
            "train_metrics": {"rmse": train_rmse, "mae": train_mae, "r2": train_r2},
            "validation_metrics": {"rmse": val_rmse, "mae": val_mae, "r2": val_r2},
            "future_predictions": future_predictions_orig,
            "annual_forecasts": annual_forecasts,
            "uncertainty": prediction_uncertainty,
            "model_summary": {
                "architecture": "Multi-layer LSTM with BatchNorm and Dropout",
                "total_parameters": model.count_params(),
                "sequence_length": self.sequence_length,
                "features_count": len(feature_cols),
                "historical_avg": historical_avg,
                "forecast_avg": np.mean(future_predictions_orig),
                "forecast_increase": np.mean(future_predictions_orig) - historical_avg,
                "training_epochs": len(history.history["loss"]),
            },
        }

        self.models["improved_lstm"] = lstm_results

        print("IMPROVED LSTM MODELING COMPLETED SUCCESSFULLY!")
        print(f"Model Performance: R²={val_r2:.4f}, RMSE={val_rmse:.4f}°C")
        print(
            f"Future warming trend: {lstm_results['model_summary']['forecast_increase']:+.2f}°C"
        )

        return lstm_results


if __name__ == "__main__":
    print("Improved LSTM Module")
    print("Enhanced LSTM implementation with proper sequence lengths")
    print("for climate data and uncertainty quantification.")
