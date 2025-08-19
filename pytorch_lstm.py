"""
PyTorch-based LSTM implementation for climate forecasting

This module provides a PyTorch replacement for the TensorFlow-based LSTM model,
designed for Python 3.13.5 compatibility and GPU acceleration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from data_dictionary import TEMPERATURE_COLUMNS
from uncertainty_quantification import UncertaintyQuantifier

# Check PyTorch availability and GPU support
PYTORCH_AVAILABLE = True
try:
    import torch

    if torch.cuda.is_available():
        print(f"PyTorch GPU support detected: {torch.cuda.get_device_name()}")
        DEVICE = torch.device("cuda")
    else:
        print("PyTorch CPU mode - no GPU detected")
        DEVICE = torch.device("cpu")
except ImportError:
    PYTORCH_AVAILABLE = False
    DEVICE = torch.device("cpu")


class ClimateLSTM(nn.Module):
    """
    Climate-specific LSTM model for temperature forecasting

    Features:
    - Multi-layer LSTM architecture with regularization
    - Batch normalization for stable training
    - Dropout for overfitting prevention
    - Climate-appropriate sequence length (365 days)
    """

    def __init__(
        self, input_size, sequence_length, hidden_sizes=None, dropout_rate=0.2
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size, hidden_sizes[0], batch_first=True, dropout=dropout_rate
        )
        self.bn1 = nn.BatchNorm1d(sequence_length)

        self.lstm2 = nn.LSTM(
            hidden_sizes[0], hidden_sizes[1], batch_first=True, dropout=dropout_rate
        )
        self.bn2 = nn.BatchNorm1d(sequence_length)

        self.lstm3 = nn.LSTM(
            hidden_sizes[1], hidden_sizes[2], batch_first=True, dropout=dropout_rate
        )
        self.bn3 = nn.BatchNorm1d(sequence_length)

        # Dense layers
        self.fc1 = nn.Linear(hidden_sizes[2], 32)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(16, 1)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)

        # First LSTM layer
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.bn1(lstm_out1)

        # Second LSTM layer
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.bn2(lstm_out2)

        # Third LSTM layer (use only the last output)
        lstm_out3, _ = self.lstm3(lstm_out2)
        lstm_out3 = self.bn3(lstm_out3)

        # Take only the last time step output
        lstm_final = lstm_out3[:, -1, :]  # (batch_size, hidden_size)

        # Dense layers
        out = self.relu(self.fc1(lstm_final))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)

        return out


class EarlyStopping:
    """Early stopping callback for PyTorch training"""

    def __init__(
        self, patience=15, min_delta=0, restore_best_weights=True, verbose=True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.wait = 0
        self.best_loss = float("inf")
        self.best_weights = None
        self.stopped_epoch = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = (
                len(model.training_history) if hasattr(model, "training_history") else 0
            )
            if self.verbose:
                print(f"Early stopping at epoch {self.stopped_epoch}")
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True

        return False


class ReduceLROnPlateau:
    """Learning rate reduction callback for PyTorch"""

    def __init__(self, optimizer, factor=0.2, patience=8, min_lr=1e-7, verbose=True):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

        self.wait = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr > self.min_lr:
                new_lr = max(current_lr * self.factor, self.min_lr)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr
                if self.verbose:
                    print(f"Reducing learning rate to {new_lr:.2e}")
                self.wait = 0


class PyTorchLSTMPredictor:
    """
    PyTorch-based LSTM predictor for climate forecasting

    Provides equivalent functionality to the TensorFlow ImprovedLSTMPredictor
    but with PyTorch backend for Python 3.13.5 compatibility.
    """

    def __init__(self, data, tree_loss_by_year=None):
        """
        Initialize PyTorch LSTM predictor with validated data

        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with climate measurements
        tree_loss_by_year : pd.DataFrame, optional
            Deforestation data
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM modeling")

        self.data = data
        self.tree_loss_by_year = tree_loss_by_year
        self.models = {}
        self.forecasts = {}
        self.scalers = {}
        self.device = DEVICE

        # Use proper temperature column
        self.temp_col = TEMPERATURE_COLUMNS["daily_mean"]
        if self.temp_col not in data.columns:
            raise ValueError(f"Temperature column {self.temp_col} not found in data")

        # Climate-appropriate sequence length (1+ years for seasonal patterns)
        self.sequence_length = 365  # 1 year of daily data for climate patterns

        print(f"Initialized PyTorch LSTM predictor with {len(data):,} records")
        print(f"Using temperature column: {self.temp_col}")
        print(f"Sequence length: {self.sequence_length} days (climate-appropriate)")
        print(f"Device: {self.device}")

    def prepare_features(self):
        """Prepare feature data for LSTM training"""
        print("Preparing features for PyTorch LSTM...")

        # Start with temperature data
        feature_data = self.data.copy()

        # Add temporal features
        feature_data["year"] = feature_data.index.year
        feature_data["month"] = feature_data.index.month
        feature_data["day_of_year"] = feature_data.index.dayofyear

        # Add seasonal features
        feature_data["sin_day"] = np.sin(
            2 * np.pi * feature_data["day_of_year"] / 365.25
        )
        feature_data["cos_day"] = np.cos(
            2 * np.pi * feature_data["day_of_year"] / 365.25
        )

        # Add deforestation data if available
        if self.tree_loss_by_year is not None:
            # Merge deforestation data
            feature_data = feature_data.merge(
                self.tree_loss_by_year[["year", "tree_loss_ha"]], on="year", how="left"
            )
            feature_data["tree_loss_ha"] = feature_data["tree_loss_ha"].fillna(0)

        # Available features (excluding target)
        available_features = [
            col
            for col in feature_data.columns
            if col != self.temp_col and not col.startswith("heatwave")
        ]

        # Filter numeric columns only
        numeric_features = []
        for col in available_features:
            if feature_data[col].dtype in ["int64", "float64"]:
                numeric_features.append(col)

        self.feature_data = feature_data
        self.available_features = numeric_features

        print(f"Features prepared: {len(numeric_features)} features")
        print(f"Features: {numeric_features}")

    def create_sequences(self, X, y, sequence_length):
        """Create sequences for LSTM training"""
        X_sequences = []
        y_sequences = []

        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i - sequence_length : i])
            y_sequences.append(y[i])

        return np.array(X_sequences), np.array(y_sequences)

    def fit_pytorch_lstm(
        self, validation_split=0.2, epochs=50, batch_size=32, learning_rate=0.001
    ):
        """
        Fit PyTorch LSTM model with proper architecture and validation

        Parameters:
        -----------
        validation_split : float
            Proportion of data for validation
        epochs : int
            Maximum training epochs
        batch_size : int
            Training batch size
        learning_rate : float
            Initial learning rate

        Returns:
        --------
        dict : Model results and predictions
        """
        print("=" * 70)
        print("PYTORCH LSTM CLIMATE FORECASTING")
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
        split_idx = int(len(X_sequences) * (1 - validation_split))
        X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_val = y_sequences[:split_idx], y_sequences[split_idx:]

        print(f"Training set: {len(X_train)} sequences")
        print(f"Validation set: {len(X_val)} sequences")

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Build PyTorch LSTM model
        print("Building PyTorch LSTM architecture...")

        input_size = X_sequences.shape[2]
        model = ClimateLSTM(
            input_size=input_size,
            sequence_length=self.sequence_length,
            hidden_sizes=[64, 32, 16],
            dropout_rate=0.2,
        ).to(self.device)

        # Set up training components
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

        # Set up callbacks
        early_stopping = EarlyStopping(
            patience=15, restore_best_weights=True, verbose=True
        )
        reduce_lr = ReduceLROnPlateau(
            optimizer, factor=0.2, patience=8, min_lr=1e-7, verbose=True
        )

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("Starting training...")

        # Training history
        history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []
            train_maes = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                # Calculate MAE
                with torch.no_grad():
                    mae = torch.mean(torch.abs(outputs.squeeze() - batch_y))
                    train_maes.append(mae.item())

            # Validation phase
            model.eval()
            val_losses = []
            val_maes = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    mae = torch.mean(torch.abs(outputs.squeeze() - batch_y))

                    val_losses.append(loss.item())
                    val_maes.append(mae.item())

            # Record history
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_mae = np.mean(train_maes)
            avg_val_mae = np.mean(val_maes)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["train_mae"].append(avg_train_mae)
            history["val_mae"].append(avg_val_mae)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                    f"Train MAE: {avg_train_mae:.6f}, Val MAE: {avg_val_mae:.6f}"
                )

            # Apply callbacks
            reduce_lr(avg_val_loss)
            if early_stopping(avg_val_loss, model):
                break

        print("PyTorch LSTM training completed!")

        # Generate predictions for evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).cpu().numpy()
            val_pred = model(X_val_tensor).cpu().numpy()

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
        last_sequence = (
            torch.FloatTensor(X_scaled[-self.sequence_length :])
            .unsqueeze(0)
            .to(self.device)
        )

        # Generate predictions for next 6 years (2190 days)
        future_days = 365 * 6  # 6 years
        future_predictions = []

        model.eval()
        with torch.no_grad():
            current_sequence = last_sequence.clone()

            for _day in range(future_days):
                # Predict next day
                next_pred = model(current_sequence)
                future_predictions.append(next_pred.item())

                # Update sequence for next prediction (simplified approach)
                next_features = current_sequence[0, -1, :].clone()
                next_features[0] = next_pred.item()  # Update temperature prediction

                # Shift sequence window
                current_sequence = torch.roll(current_sequence, -1, dims=1)
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

        print("Annual Temperature Forecasts (PyTorch LSTM):")
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
        pytorch_results = {
            "model": model,
            "scalers": self.scalers,
            "sequence_length": self.sequence_length,
            "features_used": feature_cols,
            "training_history": history,
            "train_metrics": {"rmse": train_rmse, "mae": train_mae, "r2": train_r2},
            "validation_metrics": {"rmse": val_rmse, "mae": val_mae, "r2": val_r2},
            "future_predictions": future_predictions_orig,
            "annual_forecasts": annual_forecasts,
            "uncertainty": prediction_uncertainty,
            "model_summary": {
                "architecture": "Multi-layer PyTorch LSTM with BatchNorm and Dropout",
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "sequence_length": self.sequence_length,
                "features_count": len(feature_cols),
                "historical_avg": historical_avg,
                "forecast_avg": np.mean(future_predictions_orig),
                "forecast_increase": np.mean(future_predictions_orig) - historical_avg,
                "training_epochs": len(history["train_loss"]),
                "device": str(self.device),
            },
        }

        self.models["pytorch_lstm"] = pytorch_results

        print("PYTORCH LSTM MODELING COMPLETED SUCCESSFULLY!")
        print(f"Model Performance: R²={val_r2:.4f}, RMSE={val_rmse:.4f}°C")
        print(
            f"Future warming trend: {pytorch_results['model_summary']['forecast_increase']:+.2f}°C"
        )
        print(f"Training device: {self.device}")

        return pytorch_results
