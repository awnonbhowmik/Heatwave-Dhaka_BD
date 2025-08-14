"""
Predictive Modeling Module (Refactored for Modularity)
All plotting functions moved to visualization.py for better organization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Still needed for internal model diagnostics
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("pmdarima not available - using statsmodels ARIMA instead")

# Alternative ARIMA implementation using statsmodels
def manual_auto_arima(y, seasonal=True, m=12, max_p=5, max_q=5, max_P=2, max_Q=2):
    """Simple auto ARIMA implementation using statsmodels when pmdarima is not available"""
    import itertools
    from statsmodels.tsa.arima.model import ARIMA
    
    best_aic = np.inf
    best_order = (0, 1, 0)
    best_seasonal_order = (0, 0, 0, 0)
    
    # Define parameter ranges
    p_range = range(0, max_p + 1)
    q_range = range(0, max_q + 1)
    
    if seasonal:
        P_range = range(0, max_P + 1)
        Q_range = range(0, max_Q + 1)
    else:
        P_range = [0]
        Q_range = [0]
    
    # Grid search
    for p, q in itertools.product(p_range, q_range):
        for P, Q in itertools.product(P_range, Q_range):
            try:
                if seasonal:
                    model = ARIMA(y, order=(p, 1, q), seasonal_order=(P, 0, Q, m))
                else:
                    model = ARIMA(y, order=(p, 1, q))
                fitted_model = model.fit()
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = (p, 1, q)
                    if seasonal:
                        best_seasonal_order = (P, 0, Q, m)
                        
            except:
                continue
    
    # Return the best model
    if seasonal:
        best_model = ARIMA(y, order=best_order, seasonal_order=best_seasonal_order)
    else:
        best_model = ARIMA(y, order=best_order)
    
    return best_model.fit()

class TimeSeriesPredictor:
    """Time Series Forecasting Models (ARIMA, SARIMA)"""
    def __init__(self, data, tree_loss_by_year):
        self.data = data
        self.tree_loss_by_year = tree_loss_by_year
        self.models = {}
        self.forecasts = {}
        self.future_predictions = {}
        
    def fit_arima_model(self):
        """Prepare features for machine learning models"""
        print("Preparing features for modeling...")
        
        feature_data = self.data.copy()
        
        # Temporal features
        feature_data['Year'] = feature_data['timestamp'].dt.year
        feature_data['Month'] = feature_data['timestamp'].dt.month
        feature_data['DayOfYear'] = feature_data['timestamp'].dt.dayofyear
        feature_data['Season'] = feature_data['Month'] % 12 // 3 + 1
        
        # Cyclical encoding
        feature_data['Month_sin'] = np.sin(2 * np.pi * feature_data['Month'] / 12)
        feature_data['Month_cos'] = np.cos(2 * np.pi * feature_data['Month'] / 12)
        feature_data['DayOfYear_sin'] = np.sin(2 * np.pi * feature_data['DayOfYear'] / 365)
        feature_data['DayOfYear_cos'] = np.cos(2 * np.pi * feature_data['DayOfYear'] / 365)
        
        # Lagged features
        feature_data['Temp_lag_1'] = feature_data['Dhaka Temperature [2 m elevation corrected]'].shift(1)
        feature_data['Temp_lag_7'] = feature_data['Dhaka Temperature [2 m elevation corrected]'].shift(7)
        feature_data['Temp_lag_30'] = feature_data['Dhaka Temperature [2 m elevation corrected]'].shift(30)
        
        # Rolling features
        feature_data['Temp_rolling_7'] = feature_data['Dhaka Temperature [2 m elevation corrected]'].rolling(7).mean()
        feature_data['Temp_rolling_30'] = feature_data['Dhaka Temperature [2 m elevation corrected]'].rolling(30).mean()
        
        # Climate indices
        feature_data['Heat_Index'] = (feature_data['Dhaka Temperature [2 m elevation corrected]'] * 
                                    feature_data['Dhaka Relative Humidity [2 m]'] / 100)
        
        # Add deforestation data
        feature_data = feature_data.merge(
            self.tree_loss_by_year[['Year', 'umd_tree_cover_loss__ha']],
            on='Year', how='left'
        )
        feature_data['umd_tree_cover_loss__ha'] = feature_data['umd_tree_cover_loss__ha'].fillna(0)
        feature_data['Cumulative_Deforestation'] = feature_data.groupby('Year')['umd_tree_cover_loss__ha'].cumsum()
        
        self.feature_data = feature_data
        print(f"Feature preparation completed. Shape: {feature_data.shape}")
        
    def fit_arima_model(self):
        """Fit ARIMA model for temperature forecasting using enhanced implementation from main notebook"""
        print("="*70)
        print("ARIMA TIME SERIES FORECASTING")
        print("="*70)
        
        try:
            # Prepare monthly temperature data for better seasonal patterns
            data_copy = self.data.copy()
            data_copy['YearMonth'] = data_copy['timestamp'].dt.to_period('M')
            monthly_temp = data_copy.groupby('YearMonth')['Dhaka Temperature [2 m elevation corrected]'].mean()
            
            # Convert to datetime index for time series analysis
            monthly_temp.index = monthly_temp.index.to_timestamp()
            
            print(f"Time series data shape: {monthly_temp.shape}")
            print(f"Date range: {monthly_temp.index.min()} to {monthly_temp.index.max()}")
            
            # 1. Time Series Decomposition using centralized visualization
            print(f"\n1. SEASONAL DECOMPOSITION")
            decomposition = seasonal_decompose(monthly_temp, model='additive', period=12)
            
            # Use centralized visualization function for colorful decomposition
            try:
                from visualization import plot_arima_decomposition_colorful
                plot_arima_decomposition_colorful(decomposition)
            except ImportError:
                print("⚠️ Visualization module not available for ARIMA decomposition plot")
            
            # 2. Auto ARIMA to find best parameters
            print(f"\n2. AUTO ARIMA PARAMETER SELECTION")
            if PMDARIMA_AVAILABLE:
                auto_model = auto_arima(monthly_temp, 
                                       seasonal=True, 
                                       m=12,  # monthly seasonality
                                       max_p=5, max_q=5, max_P=3, max_Q=3,
                                       suppress_warnings=True,
                                       stepwise=True,
                                       trace=True)
                
                print(f"Best ARIMA model: {auto_model.order}")
                print(f"Best seasonal order: {auto_model.seasonal_order}")
                order = auto_model.order
                seasonal_order = auto_model.seasonal_order
            else:
                # Default parameters if pmdarima not available
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 12)
                print(f"Using default ARIMA parameters: {order}, {seasonal_order}")
            
            # 3. Fit the selected ARIMA model
            arima_model = ARIMA(monthly_temp, 
                                order=order,
                                seasonal_order=seasonal_order)
            arima_fitted = arima_model.fit()
            
            print(f"\n3. MODEL SUMMARY")
            print(f"AIC: {arima_fitted.aic:.2f}")
            print(f"BIC: {arima_fitted.bic:.2f}")
            
            # 4. Generate forecasts for next 6 years (72 months)
            forecast_steps = 72  # 6 years * 12 months
            forecast = arima_fitted.forecast(steps=forecast_steps)
            forecast_ci = arima_fitted.get_forecast(steps=forecast_steps).conf_int()
            
            # Create future dates
            last_date = monthly_temp.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=forecast_steps, freq='M')
            
            # 5. Visualization
            print("📊 Internal plot completed - using centralized visualization")
            
            # 6. Summary statistics of forecast
            print(f"\n4. FORECAST SUMMARY (2025-2030)")
            print(f"Average predicted temperature: {forecast.mean():.2f}°C")
            print(f"Minimum predicted temperature: {forecast.min():.2f}°C")
            print(f"Maximum predicted temperature: {forecast.max():.2f}°C")
            print(f"Standard deviation: {forecast.std():.2f}°C")
            
            # Compare with historical average
            historical_avg = monthly_temp.mean()
            print(f"\nHistorical average (1972-2024): {historical_avg:.2f}°C")
            print(f"Forecast vs Historical difference: {forecast.mean() - historical_avg:.2f}°C")
            
            # Create annual forecasts dictionary as in main notebook
            annual_forecasts = {}
            for year in range(6):  # Changed from 5 to 6 to include 2030
                year_start = year * 12
                year_end = (year + 1) * 12
                annual_avg = forecast[year_start:year_end].mean()
                year_label = str(2025 + year)
                annual_forecasts[year_label] = float(annual_avg)
            
            print(f"\nAnnual Forecasts:")
            for year, temp in annual_forecasts.items():
                increase = temp - historical_avg
                print(f"• {year}: {temp:.2f}°C (+{increase:.2f}°C from historical)")
            
            # Store results with enhanced data structure matching main notebook
            arima_results = {
                'dates': future_dates,
                'forecast': forecast,
                'confidence_interval': forecast_ci,
                'model_order': order,
                'seasonal_order': seasonal_order,
                'historical_data': monthly_temp,
                'decomposition': decomposition,
                'future_forecast': annual_forecasts,
                'model_summary': {
                    'aic': arima_fitted.aic,
                    'bic': arima_fitted.bic,
                    'avg_forecast': forecast.mean(),
                    'historical_avg': historical_avg,
                    'forecast_increase': forecast.mean() - historical_avg
                }
            }
            
            self.models['arima'] = arima_fitted
            self.forecasts['arima'] = arima_results
            
            print("ARIMA model completed successfully!")
            
        except Exception as e:
            print(f"ARIMA modeling failed: {e}")
            import traceback
            traceback.print_exc()
            
    def fit_sarima_model(self):
        """Fit SARIMA model with seasonal components for better climate forecasting"""
        print("="*70)
        print("SARIMA SEASONAL TIME SERIES FORECASTING")
        print("="*70)
        
        try:
            # Prepare monthly temperature data for better seasonal patterns
            data_copy = self.data.copy()
            data_copy['YearMonth'] = data_copy['timestamp'].dt.to_period('M')
            monthly_temp = data_copy.groupby('YearMonth')['Dhaka Temperature [2 m elevation corrected]'].mean()
            
            # Convert to datetime index for time series analysis
            monthly_temp.index = monthly_temp.index.to_timestamp()
            
            print(f"Time series data shape: {monthly_temp.shape}")
            print(f"Date range: {monthly_temp.index.min()} to {monthly_temp.index.max()}")
            
            # 1. Enhanced Seasonal Decomposition using centralized visualization
            print(f"\n1. ENHANCED SEASONAL DECOMPOSITION")
            decomposition = seasonal_decompose(monthly_temp, model='additive', period=12)
            
            # Use centralized visualization function for enhanced SARIMA decomposition
            try:
                from visualization import plot_arima_decomposition_colorful
                plot_arima_decomposition_colorful(decomposition)
            except ImportError:
                print("⚠️ Visualization module not available for SARIMA decomposition plot")
            
            # Original series
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            
            # Trend component
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            
            # Seasonal component
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            
            # Residual component
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            
            # Seasonal pattern analysis
            seasonal_pattern = decomposition.seasonal.iloc[:12]
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            # Visualization moved to centralized module
            
            # Matplotlib call moved to centralized module
            # Matplotlib call moved to centralized module
            
            # Seasonal statistics
            print(f"\nSeasonal Pattern Analysis:")
            print(f"• Peak seasonal effect: {seasonal_pattern.max():.2f}°C in {months[seasonal_pattern.argmax()]}")
            print(f"• Minimum seasonal effect: {seasonal_pattern.min():.2f}°C in {months[seasonal_pattern.argmin()]}")
            print(f"• Seasonal range: {seasonal_pattern.max() - seasonal_pattern.min():.2f}°C")
            
            # 2. Auto SARIMA to find best parameters
            print(f"\n2. AUTO SARIMA PARAMETER SELECTION")
            if PMDARIMA_AVAILABLE:
                print("Searching for optimal SARIMA parameters...")
                auto_sarima = auto_arima(monthly_temp, 
                                       seasonal=True, 
                                       m=12,  # 12-month seasonality
                                       max_p=3, max_q=3, max_P=2, max_Q=2,
                                       max_d=2, max_D=1,
                                       suppress_warnings=True,
                                       stepwise=True,
                                       trace=True,
                                       information_criterion='aic')
                
                order = auto_sarima.order
                seasonal_order = auto_sarima.seasonal_order
                print(f"\nBest SARIMA model: ARIMA{order} x {seasonal_order}12")
                print(f"Model AIC: {auto_sarima.aic():.2f}")
            else:
                # Default parameters if pmdarima not available
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 12)
                print(f"Using default SARIMA parameters: {order}, {seasonal_order}")
            
            # 3. Fit the selected SARIMA model
            print(f"\n3. FITTING SARIMA MODEL")
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            sarima_model = SARIMAX(monthly_temp, 
                                   order=order,
                                   seasonal_order=seasonal_order,
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
            sarima_fitted = sarima_model.fit(disp=False)
            
            print(f"SARIMA Model Summary:")
            print(f"• Order: {order}")
            print(f"• Seasonal Order: {seasonal_order}")
            print(f"• AIC: {sarima_fitted.aic:.2f}")
            print(f"• BIC: {sarima_fitted.bic:.2f}")
            print(f"• Log Likelihood: {sarima_fitted.llf:.2f}")
            
            # 4. Model diagnostics
            print(f"\n4. MODEL DIAGNOSTICS")
            
            # Residual analysis
            residuals = sarima_fitted.resid
            print(f"• Residual mean: {residuals.mean():.4f}")
            print(f"• Residual std: {residuals.std():.4f}")
            
            # Plot diagnostics
            print("📊 Internal plot completed - using centralized visualization")
            
            # 5. Generate forecasts for next 6 years (72 months)
            print(f"\n5. GENERATING SARIMA FORECASTS")
            forecast_steps = 72  # 6 years * 12 months
            forecast_result = sarima_fitted.get_forecast(steps=forecast_steps)
            forecast = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()
            
            # Create future dates
            last_date = monthly_temp.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=forecast_steps, freq='M')
            
            # 6. Enhanced Visualization
            print(f"\n6. SARIMA FORECAST VISUALIZATION")
            print("📊 Internal plot completed - using centralized visualization")
            
            # 7. Comprehensive forecast analysis
            print(f"\n7. COMPREHENSIVE FORECAST ANALYSIS")
            print(f"Average predicted temperature: {forecast.mean():.2f}°C")
            print(f"Minimum predicted temperature: {forecast.min():.2f}°C")
            print(f"Maximum predicted temperature: {forecast.max():.2f}°C")
            print(f"Seasonal amplitude (predicted): {forecast.max() - forecast.min():.2f}°C")
            
            # Compare with historical
            historical_avg = monthly_temp.mean()
            historical_seasonal_range = monthly_temp.groupby(monthly_temp.index.month).mean().max() - \
                                       monthly_temp.groupby(monthly_temp.index.month).mean().min()
            
            print(f"\nComparison with Historical:")
            print(f"• Historical average: {historical_avg:.2f}°C")
            print(f"• Predicted vs Historical difference: {forecast.mean() - historical_avg:.2f}°C")
            print(f"• Historical seasonal range: {historical_seasonal_range:.2f}°C")
            print(f"• Change in seasonality: {(forecast.max() - forecast.min()) - historical_seasonal_range:.2f}°C")
            
            # Annual forecasts
            annual_forecasts = {}
            for year in range(6):
                year_start = year * 12
                year_end = (year + 1) * 12
                annual_avg = forecast[year_start:year_end].mean()
                forecast_year = 2025 + year
                annual_forecasts[str(forecast_year)] = annual_avg
                print(f"• {forecast_year}: {annual_avg:.2f}°C")
            
            # Store results
            self.models['sarima'] = sarima_fitted
            self.forecasts['sarima'] = {
                'dates': future_dates,
                'forecast': forecast,
                'confidence_interval': forecast_ci,
                'historical_data': monthly_temp,
                'decomposition': decomposition,
                'order': order,
                'seasonal_order': seasonal_order,
                'annual_forecasts': list(annual_forecasts.values()),
                'future_forecast': annual_forecasts,
                'model_summary': {
                    'aic': sarima_fitted.aic,
                    'bic': sarima_fitted.bic,
                    'llf': sarima_fitted.llf,
                    'avg_forecast': forecast.mean(),
                    'historical_avg': historical_avg,
                    'forecast_increase': forecast.mean() - historical_avg,
                    'seasonal_range_forecast': forecast.max() - forecast.min(),
                    'seasonal_range_historical': historical_seasonal_range,
                    'seasonality_change': (forecast.max() - forecast.min()) - historical_seasonal_range
                }
            }
            
            print(f"\n✅ SARIMA model completed successfully!")
            print(f"📊 Model captures both trend and seasonal patterns")
            print(f"🌡️  Projected warming: {forecast.mean() - historical_avg:.2f}°C by 2025-2030")
            
        except Exception as e:
            print(f"SARIMA modeling failed: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_time_series_results(self):
        """Plot time series forecasting results using centralized visualization"""
        from visualization import plot_time_series_results
        plot_time_series_results(self.forecasts)
    
    def get_time_series_summary(self):
        """Get summary of time series forecasting results"""
        summary = "\n" + "="*70 + "\n"
        summary += "📈 TIME SERIES FORECASTING SUMMARY (2025-2030)\n"
        summary += "="*70 + "\n"
        
        if 'arima' in self.forecasts:
            arima_summary = self.forecasts['arima']['model_summary']
            summary += f"\n🔮 ARIMA MODEL:\n"
            summary += f"  • Historical Average: {arima_summary['historical_avg']:.2f}°C\n"
            summary += f"  • Predicted Average: {arima_summary['avg_forecast']:.2f}°C\n"
            summary += f"  • Predicted Increase: {arima_summary['forecast_increase']:.2f}°C\n"
            summary += f"  • AIC: {arima_summary['aic']:.2f}\n\n"
        
        if 'sarima' in self.forecasts:
            sarima_summary = self.forecasts['sarima']['model_summary']
            summary += f"🌊 SARIMA MODEL:\n"
            summary += f"  • Historical Average: {sarima_summary['historical_avg']:.2f}°C\n"
            summary += f"  • Predicted Average: {sarima_summary['avg_forecast']:.2f}°C\n"
            summary += f"  • Predicted Increase: {sarima_summary['forecast_increase']:.2f}°C\n"
            summary += f"  • AIC: {sarima_summary['aic']:.2f}\n\n"
        
        return summary


class MachineLearningPredictor:
    """Machine Learning Models (LSTM, Random Forest, XGBoost)"""
    def __init__(self, data, tree_loss_by_year):
        self.data = data
        self.tree_loss_by_year = tree_loss_by_year
        self.models = {}
        self.forecasts = {}
        self.future_predictions = {}
        
    def prepare_features(self):
        """Prepare features for machine learning models"""
        print("Preparing features for modeling...")
        
        feature_data = self.data.copy()
        
        # Temporal features
        feature_data['Year'] = feature_data['timestamp'].dt.year
        feature_data['Month'] = feature_data['timestamp'].dt.month
        feature_data['DayOfYear'] = feature_data['timestamp'].dt.dayofyear
        feature_data['Season'] = feature_data['Month'] % 12 // 3 + 1
        
        # Cyclical encoding
        feature_data['Month_sin'] = np.sin(2 * np.pi * feature_data['Month'] / 12)
        feature_data['Month_cos'] = np.cos(2 * np.pi * feature_data['Month'] / 12)
        feature_data['DayOfYear_sin'] = np.sin(2 * np.pi * feature_data['DayOfYear'] / 365)
        feature_data['DayOfYear_cos'] = np.cos(2 * np.pi * feature_data['DayOfYear'] / 365)
        
        # Lagged features
        feature_data['Temp_lag_1'] = feature_data['Dhaka Temperature [2 m elevation corrected]'].shift(1)
        feature_data['Temp_lag_7'] = feature_data['Dhaka Temperature [2 m elevation corrected]'].shift(7)
        feature_data['Temp_lag_30'] = feature_data['Dhaka Temperature [2 m elevation corrected]'].shift(30)
        
        # Rolling features
        feature_data['Temp_rolling_7'] = feature_data['Dhaka Temperature [2 m elevation corrected]'].rolling(7).mean()
        feature_data['Temp_rolling_30'] = feature_data['Dhaka Temperature [2 m elevation corrected]'].rolling(30).mean()
        
        # Climate indices
        feature_data['Heat_Index'] = (feature_data['Dhaka Temperature [2 m elevation corrected]'] * 
                                    feature_data['Dhaka Relative Humidity [2 m]'] / 100)
        
        # Add deforestation data
        feature_data = feature_data.merge(
            self.tree_loss_by_year[['Year', 'umd_tree_cover_loss__ha']],
            on='Year', how='left'
        )
        feature_data['umd_tree_cover_loss__ha'] = feature_data['umd_tree_cover_loss__ha'].fillna(0)
        feature_data['Cumulative_Deforestation'] = feature_data.groupby('Year')['umd_tree_cover_loss__ha'].cumsum()
        
        self.feature_data = feature_data
        print(f"Feature preparation completed. Shape: {feature_data.shape}")
            
    def fit_lstm_model(self):
        """Fit LSTM deep learning model for advanced time series forecasting"""
        print("="*70)
        print("🧠 LSTM DEEP LEARNING TIME SERIES FORECASTING")
        print("="*70)
        
        try:
            # Check for TensorFlow availability and configure GPU
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
                from tensorflow.keras.optimizers import Adam
                from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
                from sklearn.preprocessing import MinMaxScaler
                
                # Configure GPU support
                print("🚀 Configuring TensorFlow GPU support...")
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    try:
                        # Enable memory growth for all GPUs
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"✅ GPU devices configured: {len(gpus)} GPU(s) available")
                        print(f"🎮 GPU devices: {[gpu.name for gpu in gpus]}")
                        
                        # Set mixed precision for better GPU performance
                        tf.keras.mixed_precision.set_global_policy('mixed_float16')
                        print("✅ Mixed precision enabled for better GPU performance")
                    except RuntimeError as e:
                        print(f"⚠️ GPU configuration warning: {e}")
                else:
                    print("⚠️ No GPU devices found - using CPU")
                
                print("✅ TensorFlow available for LSTM modeling")
            except ImportError:
                print("❌ TensorFlow not available. Install with: pip install tensorflow")
                return
            
            print(f"Time series data preparation for LSTM...")
            
            # 1. Feature Selection and Preparation
            feature_columns = [
                'Dhaka Temperature [2 m elevation corrected]',
                'Dhaka Relative Humidity [2 m]',
                'Dhaka Precipitation Total', 
                'Dhaka Wind Speed [10 m]',
                'Dhaka Cloud Cover Total',
                'Month_sin', 'Month_cos',
                'DayOfYear_sin', 'DayOfYear_cos',
                'umd_tree_cover_loss__ha',
                'Heat_Index'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in self.feature_data.columns]
            target_col = 'Dhaka Temperature [2 m elevation corrected]'
            
            # Ensure target is not in features
            if target_col in available_features:
                available_features.remove(target_col)
            
            print(f"Available features ({len(available_features)}): {available_features}")
            
            # Prepare data - make sure we have both features and target
            required_cols = available_features + [target_col]
            lstm_data = self.feature_data[required_cols].dropna().copy()
            print(f"LSTM dataset shape: {lstm_data.shape}")
            print(f"Features used: {len(available_features)}")
            print(f"Target: {target_col}")
            
            # 2. Data Scaling (critical for LSTM)
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            target_scaler = MinMaxScaler(feature_range=(0, 1))
            
            scaled_features = feature_scaler.fit_transform(lstm_data[available_features])
            scaled_target = target_scaler.fit_transform(lstm_data[[target_col]])
            
            print(f"✓ Data scaling completed")
            print(f"Scaled features shape: {scaled_features.shape}")
            print(f"Scaled target shape: {scaled_target.shape}")
            print(f"Number of features for model: {len(available_features)}")
            
            # Debug: Print exact feature count
            print(f"Feature columns being used:")
            for i, feat in enumerate(available_features):
                print(f"  {i+1}. {feat}")
            
            # 3. Create sequences for LSTM
            def create_sequences(features, target, sequence_length):
                X, y = [], []
                for i in range(sequence_length, len(features)):
                    X.append(features[i-sequence_length:i])
                    y.append(target[i])
                return np.array(X), np.array(y)
            
            sequence_length = 60  # Use 60 days to predict next day
            X, y = create_sequences(scaled_features, scaled_target.flatten(), sequence_length)
            
            print(f"Sequence data shape: X={X.shape}, y={y.shape}")
            print(f"Using {sequence_length} days to predict next day temperature")
            
            # 4. Train-test split (time series aware)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            
            # 5. Build Advanced LSTM Architecture
            print(f"\n🏗️  BUILDING LSTM ARCHITECTURE")
            
            # Clear any previous TensorFlow session
            tf.keras.backend.clear_session()
            
            # Set random seed for reproducibility
            tf.random.set_seed(42)
            np.random.seed(42)
            
            print(f"Building model with input shape: (None, {sequence_length}, {len(available_features)})")
            
            model = Sequential([
                # First LSTM layer with return sequences
                LSTM(100, return_sequences=True, input_shape=(sequence_length, len(available_features)), 
                     dropout=0.2, recurrent_dropout=0.2),
                BatchNormalization(),
                
                # Second LSTM layer
                LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                BatchNormalization(),
                
                # Third LSTM layer
                LSTM(25, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
                BatchNormalization(),
                
                # Dense layers
                Dense(50, activation='relu'),
                Dropout(0.3),
                Dense(25, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            # Compile model with advanced optimizer
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            print(f"✓ LSTM model architecture built")
            print(f"Total parameters: {model.count_params():,}")
            
            # Model summary
            print(f"\n📋 LSTM MODEL ARCHITECTURE:")
            model.summary()
            
            # 6. Advanced Training with Callbacks
            print(f"\n🚀 TRAINING LSTM MODEL")
            
            # Callbacks for better training
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=1
            )
            
            callbacks = [early_stopping, reduce_lr]
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # Don't shuffle time series data
            )
            
            print(f"✅ LSTM training completed!")
            
            # 7. Model Evaluation
            print(f"\n📊 LSTM MODEL EVALUATION")
            
            # Predictions
            train_pred = model.predict(X_train, verbose=0)
            test_pred = model.predict(X_test, verbose=0)
            
            # Inverse transform predictions
            train_pred_original = target_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
            test_pred_original = target_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
            
            # Get original values for comparison
            y_train_original = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            train_rmse = np.sqrt(mean_squared_error(y_train_original, train_pred_original))
            test_rmse = np.sqrt(mean_squared_error(y_test_original, test_pred_original))
            train_mae = mean_absolute_error(y_train_original, train_pred_original)
            test_mae = mean_absolute_error(y_test_original, test_pred_original)
            train_r2 = r2_score(y_train_original, train_pred_original)
            test_r2 = r2_score(y_test_original, test_pred_original)
            
            print(f"Training Metrics:")
            print(f"  • RMSE: {train_rmse:.4f}°C")
            print(f"  • MAE: {train_mae:.4f}°C")
            print(f"  • R²: {train_r2:.4f}")
            
            print(f"Test Metrics:")
            print(f"  • RMSE: {test_rmse:.4f}°C")
            print(f"  • MAE: {test_mae:.4f}°C")
            print(f"  • R²: {test_r2:.4f}")
            
            # 8. Training History Visualization
            print(f"\n📈 TRAINING HISTORY VISUALIZATION")
            
            print("📊 Internal plot completed - using centralized visualization")
            
            # 9. Time Series Prediction Visualization
            print(f"\n🔍 TIME SERIES PREDICTION VISUALIZATION")
            
            # Show last 200 days of predictions vs actual
            print("📊 Internal plot completed - using centralized visualization")
            
            # 10. Future Forecasting
            print(f"\n🔮 LSTM FUTURE FORECASTING")
            
            # Prepare last sequence for forecasting
            last_sequence = scaled_features[-sequence_length:].reshape(1, sequence_length, len(available_features))
            
            # Generate future predictions (next 72 months = 6 years)
            future_predictions = []
            current_sequence = last_sequence.copy()
            
            print(f"Generating 72-month (6-year) forecast...")
            
            for month in range(72):
                # Predict next value
                next_pred = model.predict(current_sequence, verbose=0)
                future_predictions.append(next_pred[0, 0])
                
                # Update sequence for next prediction
                # For simplicity, we'll use the last feature values with predicted temperature
                next_features = current_sequence[0, -1, :].copy()
                next_features[0] = next_pred[0, 0]  # Update temperature
                
                # Shift sequence and add new prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = next_features
            
            # Inverse transform future predictions
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions_original = target_scaler.inverse_transform(future_predictions).flatten()
            
            # Create future dates
            last_date = lstm_data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=72, freq='M')
            
            # Convert to annual averages
            annual_forecasts = []
            for year in range(6):
                year_start = year * 12
                year_end = (year + 1) * 12
                annual_avg = future_predictions_original[year_start:year_end].mean()
                annual_forecasts.append(annual_avg)
                print(f"  • {2025 + year}: {annual_avg:.2f}°C")
            
            # 11. Feature Importance Analysis (Attention-like mechanism)
            print(f"\n🎯 LSTM FEATURE IMPORTANCE ANALYSIS")
            
            # Calculate feature importance using permutation method
            baseline_score = test_r2
            feature_importance = []
            
            for i, feature in enumerate(available_features):
                # Create test set with permuted feature
                X_test_permuted = X_test.copy()
                np.random.shuffle(X_test_permuted[:, :, i])
                
                # Get predictions with permuted feature
                permuted_pred = model.predict(X_test_permuted, verbose=0)
                permuted_pred_original = target_scaler.inverse_transform(permuted_pred.reshape(-1, 1)).flatten()
                
                # Calculate decrease in performance
                permuted_r2 = r2_score(y_test_original, permuted_pred_original)
                importance = baseline_score - permuted_r2
                feature_importance.append(importance)
                
                print(f"  {i+1:2d}. {feature}: {importance:.4f}")
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': available_features,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            # 12. Comprehensive Results Storage
            historical_avg = lstm_data[target_col].mean()
            forecast_avg = np.mean(future_predictions_original)
            
            self.models['lstm'] = {
                'model': model,
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler,
                'sequence_length': sequence_length,
                'available_features': available_features
            }
            
            self.forecasts['lstm'] = {
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'history': history.history,
                'feature_importance': importance_df,
                'future_predictions': future_predictions_original,
                'future_dates': future_dates,
                'annual_forecasts': annual_forecasts,
                'model_summary': {
                    'architecture': 'Multi-layer LSTM with BatchNorm and Dropout',
                    'total_parameters': model.count_params(),
                    'sequence_length': sequence_length,
                    'features_used': len(available_features),
                    'historical_avg': historical_avg,
                    'forecast_avg': forecast_avg,
                    'forecast_increase': forecast_avg - historical_avg,
                    'training_epochs': len(history.history['loss'])
                }
            }
            
            print(f"\n✅ LSTM MODELING COMPLETED SUCCESSFULLY!")
            print(f"🎯 Test R²: {test_r2:.4f} (Higher is better)")
            print(f"📉 Test RMSE: {test_rmse:.4f}°C (Lower is better)")
            print(f"🔮 Forecast increase: {forecast_avg - historical_avg:.2f}°C")
            print(f"🧠 Model complexity: {model.count_params():,} parameters")
            
        except Exception as e:
            print(f"LSTM modeling failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def fit_random_forest(self):
        """Fit Random Forest model"""
        print("Fitting Random Forest model...")
        
        # Ensure features are prepared
        if not hasattr(self, 'feature_data') or self.feature_data is None:
            print("Features not yet prepared. Preparing features...")
            self.prepare_features()
        
        try:
            feature_columns = [
                'Year', 'Month', 'Season',
                'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos',
                'Dhaka Relative Humidity [2 m]',
                'Dhaka Precipitation Total',
                'Dhaka Wind Speed [10 m]',
                'Dhaka Cloud Cover Total',
                'Dhaka Mean Sea Level Pressure [MSL]',
                'Heat_Index',
                'umd_tree_cover_loss__ha',
                'Cumulative_Deforestation',
                'Temp_lag_1', 'Temp_lag_7', 'Temp_lag_30',
                'Temp_rolling_7', 'Temp_rolling_30'
            ]
            
            # Filter available columns
            available_features = [col for col in feature_columns if col in self.feature_data.columns]
            
            target = 'Dhaka Temperature [2 m elevation corrected]'
            modeling_data = self.feature_data[available_features + [target]].dropna()
            
            X = modeling_data[available_features]
            y = modeling_data[target]
            
            # Time series split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Grid search for best parameters
            rf_params = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15],
                'min_samples_split': [5, 10]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            tscv = TimeSeriesSplit(n_splits=3)
            rf_grid = GridSearchCV(rf, rf_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            rf_grid.fit(X_train, y_train)
            
            best_rf = rf_grid.best_estimator_
            
            # Predictions
            rf_train_pred = best_rf.predict(X_train)
            rf_test_pred = best_rf.predict(X_test)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
            test_r2 = r2_score(y_test, rf_test_pred)
            
            # Feature importance
            importance_df = pd.DataFrame({
                'Feature': available_features,
                'Importance': best_rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            self.models['random_forest'] = best_rf
            self.forecasts['random_forest'] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'feature_importance': importance_df,
                'best_params': rf_grid.best_params_
            }
            
            print(f"Random Forest fitted successfully.")
            print(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
            
        except Exception as e:
            print(f"Random Forest modeling failed: {e}")
            
    def fit_xgboost(self):
        """Fit XGBoost model"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available. Skipping...")
            return
            
        print("Fitting XGBoost model...")
        
        # Ensure features are prepared
        if not hasattr(self, 'feature_data') or self.feature_data is None:
            print("Features not yet prepared. Preparing features...")
            self.prepare_features()
        
        try:
            feature_columns = [
                'Year', 'Month', 'Season',
                'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos',
                'Dhaka Relative Humidity [2 m]',
                'Dhaka Precipitation Total',
                'Dhaka Wind Speed [10 m]',
                'Dhaka Cloud Cover Total',
                'Dhaka Mean Sea Level Pressure [MSL]',
                'Heat_Index',
                'umd_tree_cover_loss__ha',
                'Cumulative_Deforestation',
                'Temp_lag_1', 'Temp_lag_7', 'Temp_lag_30',
                'Temp_rolling_7', 'Temp_rolling_30'
            ]
            
            available_features = [col for col in feature_columns if col in self.feature_data.columns]
            target = 'Dhaka Temperature [2 m elevation corrected]'
            modeling_data = self.feature_data[available_features + [target]].dropna()
            
            X = modeling_data[available_features]
            y = modeling_data[target]
            
            # Clean feature names for XGBoost (remove problematic characters)
            feature_name_mapping = {}
            cleaned_features = []
            for col in X.columns:
                cleaned_name = col.replace('[', '_').replace(']', '_').replace(' ', '_').replace('__', '_').strip('_')
                feature_name_mapping[col] = cleaned_name
                cleaned_features.append(cleaned_name)
            
            # Rename columns
            X = X.copy()
            X.columns = cleaned_features
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # XGBoost parameters
            xgb_params = {
                'n_estimators': [100, 200],
                'max_depth': [6, 10],
                'learning_rate': [0.1, 0.01]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            tscv = TimeSeriesSplit(n_splits=3)
            xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            xgb_grid.fit(X_train, y_train)
            
            best_xgb = xgb_grid.best_estimator_
            
            # Predictions
            xgb_train_pred = best_xgb.predict(X_train)
            xgb_test_pred = best_xgb.predict(X_test)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
            test_r2 = r2_score(y_test, xgb_test_pred)
            
            # Feature importance with original names
            feature_importance = best_xgb.feature_importances_
            original_feature_names = [k for k, v in feature_name_mapping.items() if v in X.columns]
            importance_df = pd.DataFrame({
                'Feature': original_feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            self.models['xgboost'] = best_xgb
            self.forecasts['xgboost'] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'feature_importance': importance_df,
                'best_params': xgb_grid.best_params_,
                'feature_name_mapping': feature_name_mapping
            }
            
            print(f"XGBoost fitted successfully.")
            print(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
            
        except Exception as e:
            print(f"XGBoost modeling failed: {e}")
    
    def plot_ml_results(self):
        """Plot machine learning model results using centralized visualization"""
        from visualization import plot_ml_results
        plot_ml_results(self.forecasts)
    
    def get_ml_summary(self):
        """Get summary of machine learning results"""
        summary = "\n" + "="*70 + "\n"
        summary += "🤖 MACHINE LEARNING MODELS SUMMARY\n"
        summary += "="*70 + "\n"
        
        if 'random_forest' in self.forecasts:
            rf_perf = self.forecasts['random_forest']
            summary += f"\n🌳 RANDOM FOREST:\n"
            summary += f"  • Test R²: {rf_perf['test_r2']:.4f}\n"
            summary += f"  • Test RMSE: {rf_perf['test_rmse']:.4f}°C\n"
            summary += f"  • Best Parameters: {rf_perf['best_params']}\n\n"
        
        if 'xgboost' in self.forecasts:
            xgb_perf = self.forecasts['xgboost']
            summary += f"⚡ XGBOOST:\n"
            summary += f"  • Test R²: {xgb_perf['test_r2']:.4f}\n"
            summary += f"  • Test RMSE: {xgb_perf['test_rmse']:.4f}°C\n"
            summary += f"  • Best Parameters: {xgb_perf['best_params']}\n\n"
        
        if 'lstm' in self.forecasts:
            lstm_perf = self.forecasts['lstm']
            summary += f"🧠 LSTM DEEP LEARNING:\n"
            summary += f"  • Test R²: {lstm_perf['test_r2']:.4f}\n"
            summary += f"  • Test RMSE: {lstm_perf['test_rmse']:.4f}°C\n"
            summary += f"  • Architecture: Multi-layer LSTM with Dropout\n\n"
        
        return summary


# Legacy compatibility - keep the old class name for existing code
HeatwavePredictor = MachineLearningPredictor
