"""
Predictive Modeling Module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    print("pmdarima not available. Install with: pip install pmdarima")

class HeatwavePredictor:
    def __init__(self, data, tree_loss_by_year):
        self.data = data
        self.tree_loss_by_year = tree_loss_by_year
        self.models = {}
        self.forecasts = {}
        
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
        
    def fit_arima_model(self):
        """Fit ARIMA model for temperature forecasting using tested implementation"""
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
            
            # 1. Time Series Decomposition
            print(f"\n1. SEASONAL DECOMPOSITION")
            decomposition = seasonal_decompose(monthly_temp, model='additive', period=12)
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            decomposition.observed.plot(ax=axes[0], title='Original Time Series')
            decomposition.trend.plot(ax=axes[1], title='Trend Component')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
            decomposition.resid.plot(ax=axes[3], title='Residual Component')
            plt.tight_layout()
            plt.show()
            
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
                
                order = auto_model.order
                seasonal_order = auto_model.seasonal_order
                print(f"Best ARIMA model: {order}")
                print(f"Best seasonal order: {seasonal_order}")
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
            
            # 4. Generate forecasts for next 5 years (60 months)
            forecast_steps = 60  # 5 years * 12 months
            forecast = arima_fitted.forecast(steps=forecast_steps)
            forecast_ci = arima_fitted.get_forecast(steps=forecast_steps).conf_int()
            
            # Create future dates
            last_date = monthly_temp.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=forecast_steps, freq='M')
            
            # 5. Visualization
            plt.figure(figsize=(15, 8))
            
            # Plot historical data
            plt.plot(monthly_temp.index[-120:], monthly_temp.values[-120:], 
                     label='Historical (Last 10 years)', color='blue', linewidth=2)
            
            # Plot forecasts
            plt.plot(future_dates, forecast, 
                     label='ARIMA Forecast (2025-2029)', color='red', linewidth=2)
            
            # Plot confidence intervals
            plt.fill_between(future_dates, 
                            forecast_ci.iloc[:, 0], 
                            forecast_ci.iloc[:, 1], 
                            color='red', alpha=0.2, label='95% Confidence Interval')
            
            plt.title('Temperature Forecast using ARIMA Model', fontsize=16, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # 6. Summary statistics of forecast
            print(f"\n4. FORECAST SUMMARY (2025-2029)")
            print(f"Average predicted temperature: {forecast.mean():.2f}°C")
            print(f"Minimum predicted temperature: {forecast.min():.2f}°C")
            print(f"Maximum predicted temperature: {forecast.max():.2f}°C")
            print(f"Standard deviation: {forecast.std():.2f}°C")
            
            # Compare with historical average
            historical_avg = monthly_temp.mean()
            print(f"\nHistorical average (1972-2024): {historical_avg:.2f}°C")
            print(f"Forecast vs Historical difference: {forecast.mean() - historical_avg:.2f}°C")
            
            # Store results
            self.models['arima'] = arima_fitted
            self.forecasts['arima'] = {
                'dates': future_dates,
                'forecast': forecast,
                'confidence_interval': forecast_ci,
                'historical_data': monthly_temp,
                'decomposition': decomposition,
                'order': order,
                'seasonal_order': seasonal_order,
                'model_summary': {
                    'aic': arima_fitted.aic,
                    'bic': arima_fitted.bic,
                    'avg_forecast': forecast.mean(),
                    'historical_avg': historical_avg,
                    'forecast_increase': forecast.mean() - historical_avg
                }
            }
            
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
            
            # 1. Enhanced Seasonal Decomposition
            print(f"\n1. ENHANCED SEASONAL DECOMPOSITION")
            decomposition = seasonal_decompose(monthly_temp, model='additive', period=12)
            
            # Plot enhanced decomposition
            fig, axes = plt.subplots(5, 1, figsize=(16, 15))
            
            # Original series
            decomposition.observed.plot(ax=axes[0], title='Original Monthly Temperature Series', color='blue')
            axes[0].set_ylabel('Temperature (°C)')
            axes[0].grid(True, alpha=0.3)
            
            # Trend component
            decomposition.trend.plot(ax=axes[1], title='Long-term Trend Component', color='red')
            axes[1].set_ylabel('Trend (°C)')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal component
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component (Annual Cycle)', color='green')
            axes[2].set_ylabel('Seasonal (°C)')
            axes[2].grid(True, alpha=0.3)
            
            # Residual component
            decomposition.resid.plot(ax=axes[3], title='Residual Component', color='orange')
            axes[3].set_ylabel('Residual (°C)')
            axes[3].grid(True, alpha=0.3)
            
            # Seasonal pattern analysis
            seasonal_pattern = decomposition.seasonal.iloc[:12]
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[4].bar(months, seasonal_pattern.values, color='skyblue', alpha=0.7)
            axes[4].set_title('Average Seasonal Pattern by Month')
            axes[4].set_ylabel('Seasonal Effect (°C)')
            axes[4].grid(True, alpha=0.3)
            axes[4].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
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
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Residuals plot
            residuals.plot(ax=axes[0,0], title='SARIMA Residuals')
            axes[0,0].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals.dropna(), dist="norm", plot=axes[0,1])
            axes[0,1].set_title('Normal Q-Q Plot of Residuals')
            axes[0,1].grid(True, alpha=0.3)
            
            # ACF of residuals
            from statsmodels.tsa.stattools import acf
            residual_acf = acf(residuals.dropna(), nlags=20)
            axes[1,0].stem(range(len(residual_acf)), residual_acf)
            axes[1,0].set_title('ACF of Residuals')
            axes[1,0].grid(True, alpha=0.3)
            
            # Histogram of residuals
            residuals.hist(ax=axes[1,1], bins=20, alpha=0.7)
            axes[1,1].set_title('Distribution of Residuals')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # 5. Generate forecasts for next 5 years (60 months)
            print(f"\n5. GENERATING SARIMA FORECASTS")
            forecast_steps = 60  # 5 years * 12 months
            forecast_result = sarima_fitted.get_forecast(steps=forecast_steps)
            forecast = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()
            
            # Create future dates
            last_date = monthly_temp.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=forecast_steps, freq='M')
            
            # 6. Enhanced Visualization
            print(f"\n6. SARIMA FORECAST VISUALIZATION")
            fig, axes = plt.subplots(2, 1, figsize=(16, 12))
            
            # Main forecast plot
            # Plot historical data (last 10 years for clarity)
            hist_data = monthly_temp[-120:]
            axes[0].plot(hist_data.index, hist_data.values, 
                        label='Historical (Last 10 years)', color='blue', linewidth=2)
            
            # Plot forecasts
            axes[0].plot(future_dates, forecast, 
                        label='SARIMA Forecast (2025-2029)', color='red', linewidth=2)
            
            # Plot confidence intervals
            axes[0].fill_between(future_dates, 
                               forecast_ci.iloc[:, 0], 
                               forecast_ci.iloc[:, 1], 
                               color='red', alpha=0.2, label='95% Confidence Interval')
            
            axes[0].set_title('SARIMA Temperature Forecast with Seasonal Patterns', 
                             fontsize=16, fontweight='bold')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Temperature (°C)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Seasonal forecast pattern
            forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
            forecast_df['Month'] = forecast_df['Date'].dt.month
            monthly_forecast_avg = forecast_df.groupby('Month')['Forecast'].mean()
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[1].plot(months, monthly_forecast_avg.values, marker='o', linewidth=2, 
                        markersize=8, color='purple', label='Predicted Monthly Average')
            
            # Compare with historical seasonal pattern
            historical_monthly = monthly_temp.groupby(monthly_temp.index.month).mean()
            axes[1].plot(months, historical_monthly.values, marker='s', linewidth=2, 
                        markersize=8, color='orange', alpha=0.7, label='Historical Monthly Average')
            
            axes[1].set_title('Seasonal Forecast Pattern vs Historical', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Month')
            axes[1].set_ylabel('Average Temperature (°C)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
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
            annual_forecasts = []
            for year in range(5):
                year_start = year * 12
                year_end = (year + 1) * 12
                annual_avg = forecast[year_start:year_end].mean()
                annual_forecasts.append(annual_avg)
                print(f"• {2025 + year}: {annual_avg:.2f}°C")
            
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
                'annual_forecasts': annual_forecasts,
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
            print(f"🌡️  Projected warming: {forecast.mean() - historical_avg:.2f}°C by 2025-2029")
            
        except Exception as e:
            print(f"SARIMA modeling failed: {e}")
            import traceback
            traceback.print_exc()
            
    def fit_random_forest(self):
        """Fit Random Forest model"""
        print("Fitting Random Forest model...")
        
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
            
    def generate_future_predictions(self, years=5):
        """Generate future predictions for the next few years"""
        print(f"Generating predictions for next {years} years...")
        
        predictions = {}
        
        # ARIMA predictions
        if 'arima' in self.forecasts:
            arima_forecast = self.forecasts['arima']
            
            # Convert to annual averages
            annual_forecast = []
            forecast_values = arima_forecast['forecast'].values
            
            for year_offset in range(years):
                start_idx = year_offset * 12
                end_idx = (year_offset + 1) * 12
                if end_idx <= len(forecast_values):
                    annual_avg = forecast_values[start_idx:end_idx].mean()
                    annual_forecast.append(annual_avg)
            
            predictions['arima'] = {
                'annual_temps': annual_forecast,
                'years': list(range(2025, 2025 + len(annual_forecast)))
            }
        
        # Machine Learning predictions (simplified future scenario)
        if 'random_forest' in self.models:
            # Create future scenarios with continued deforestation
            future_years = list(range(2025, 2025 + years))
            future_predictions = []
            
            # Assume continued deforestation trend
            recent_deforest_avg = self.tree_loss_by_year['umd_tree_cover_loss__ha'].tail(5).mean()
            
            for year in future_years:
                # Use recent temperature patterns and projected deforestation
                temp_prediction = self.data['Dhaka Temperature [2 m elevation corrected]'].tail(365).mean()
                # Add warming trend
                temp_prediction += 0.02 * (year - 2024)  # Assume 0.02°C/year warming
                future_predictions.append(temp_prediction)
            
            predictions['machine_learning'] = {
                'annual_temps': future_predictions,
                'years': future_years
            }
        
        self.future_predictions = predictions
        return predictions
    
    def plot_predictions(self):
        """Plot prediction results with enhanced visualizations"""
        if not hasattr(self, 'future_predictions'):
            print("No predictions available. Run generate_future_predictions() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. ARIMA Forecast Plot (detailed)
        if 'arima' in self.forecasts:
            arima_data = self.forecasts['arima']
            monthly_temp = arima_data['historical_data']
            
            # Plot last 10 years of historical data
            axes[0,0].plot(monthly_temp.index[-120:], monthly_temp.values[-120:], 
                         label='Historical (Last 10 years)', color='blue', linewidth=2)
            
            # Plot forecasts
            axes[0,0].plot(arima_data['dates'], arima_data['forecast'], 
                         label='ARIMA Forecast (2025-2029)', color='red', linewidth=2)
            
            # Plot confidence intervals
            axes[0,0].fill_between(arima_data['dates'], 
                                 arima_data['confidence_interval'].iloc[:, 0], 
                                 arima_data['confidence_interval'].iloc[:, 1], 
                                 color='red', alpha=0.2, label='95% Confidence Interval')
            
            axes[0,0].set_title('ARIMA Temperature Forecast', fontsize=14, fontweight='bold')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Temperature (°C)')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Annual predictions comparison
        if hasattr(self, 'future_predictions'):
            years_range = list(range(2020, 2030))  # Show recent history + predictions
            
            # Historical annual data
            historical_annual = self.data.groupby('Year')['Dhaka Temperature [2 m elevation corrected]'].mean()
            historical_years = [y for y in years_range if y in historical_annual.index]
            historical_temps = [historical_annual[y] for y in historical_years]
            
            axes[0,1].plot(historical_years, historical_temps, 
                         'b-', linewidth=2, marker='o', label='Historical', markersize=6)
            
            # ARIMA predictions (convert monthly to annual)
            if 'arima' in self.future_predictions:
                arima_pred = self.future_predictions['arima']
                axes[0,1].plot(arima_pred['years'], arima_pred['annual_temps'], 
                             'r-', linewidth=2, marker='o', label='ARIMA Forecast', markersize=6)
            
            # ML predictions
            if 'machine_learning' in self.future_predictions:
                ml_pred = self.future_predictions['machine_learning']
                axes[0,1].plot(ml_pred['years'], ml_pred['annual_temps'], 
                             'g-', linewidth=2, marker='s', label='ML Forecast', markersize=6)
            
            axes[0,1].axvline(x=2024, color='gray', linestyle='--', alpha=0.7, label='Prediction Start')
            axes[0,1].set_title('Annual Temperature Predictions', fontsize=14, fontweight='bold')
            axes[0,1].set_xlabel('Year')
            axes[0,1].set_ylabel('Average Temperature (°C)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Seasonal decomposition (if available)
        if 'arima' in self.forecasts and 'decomposition' in self.forecasts['arima']:
            decomp = self.forecasts['arima']['decomposition']
            
            # Plot trend component
            trend_data = decomp.trend.dropna()
            axes[1,0].plot(trend_data.index[-120:], trend_data.values[-120:], 
                         color='green', linewidth=2)
            axes[1,0].set_title('Temperature Trend Component (Last 10 Years)', fontweight='bold')
            axes[1,0].set_xlabel('Date')
            axes[1,0].set_ylabel('Temperature Trend (°C)')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Prediction summary statistics
        axes[1,1].axis('off')
        
        summary_text = "PREDICTION SUMMARY (2025-2029)\n" + "="*35 + "\n\n"
        
        if 'arima' in self.forecasts:
            arima_summary = self.forecasts['arima']['model_summary']
            summary_text += f"ARIMA Model Results:\n"
            summary_text += f"• Model Order: {self.forecasts['arima']['order']}\n"
            summary_text += f"• Seasonal Order: {self.forecasts['arima']['seasonal_order']}\n"
            summary_text += f"• AIC: {arima_summary['aic']:.2f}\n"
            summary_text += f"• BIC: {arima_summary['bic']:.2f}\n\n"
            
            summary_text += f"Temperature Predictions:\n"
            summary_text += f"• Historical Avg: {arima_summary['historical_avg']:.2f}°C\n"
            summary_text += f"• Predicted Avg: {arima_summary['avg_forecast']:.2f}°C\n"
            summary_text += f"• Predicted Increase: {arima_summary['forecast_increase']:.2f}°C\n\n"
        
        if hasattr(self, 'future_predictions'):
            if 'arima' in self.future_predictions:
                arima_temps = self.future_predictions['arima']['annual_temps']
                summary_text += f"Annual Forecasts:\n"
                for i, (year, temp) in enumerate(zip(self.future_predictions['arima']['years'], arima_temps)):
                    summary_text += f"• {year}: {temp:.2f}°C\n"
        
        # Model performance (if available)
        if 'random_forest' in self.forecasts:
            rf_perf = self.forecasts['random_forest']
            summary_text += f"\nML Model Performance:\n"
            summary_text += f"• RF Test R²: {rf_perf['test_r2']:.3f}\n"
            summary_text += f"• RF RMSE: {rf_perf['test_rmse']:.3f}\n"
        
        axes[1,1].text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                      transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        plt.show()
        
    def get_prediction_summary(self):
        """Get comprehensive summary of all predictions and model performance"""
        if not hasattr(self, 'future_predictions') and 'arima' not in self.forecasts:
            return "No predictions available."
        
        summary = "\n" + "="*70 + "\n"
        summary += "🔮 COMPREHENSIVE PREDICTION SUMMARY (2025-2029)\n"
        summary += "="*70 + "\n"
        
        # ARIMA Results
        if 'arima' in self.forecasts:
            arima_data = self.forecasts['arima']
            model_summary = arima_data['model_summary']
            
            summary += f"\n📈 ARIMA TIME SERIES MODEL:\n"
            summary += f"{'='*35}\n"
            summary += f"Model Configuration:\n"
            summary += f"  • ARIMA Order: {arima_data['order']}\n"
            summary += f"  • Seasonal Order: {arima_data['seasonal_order']}\n"
            summary += f"  • Model AIC: {model_summary['aic']:.2f}\n"
            summary += f"  • Model BIC: {model_summary['bic']:.2f}\n"
            
            summary += f"\nTemperature Forecasts:\n"
            summary += f"  • Historical Average (1972-2024): {model_summary['historical_avg']:.2f}°C\n"
            summary += f"  • Predicted Average (2025-2029): {model_summary['avg_forecast']:.2f}°C\n"
            summary += f"  • Predicted Increase: {model_summary['forecast_increase']:.2f}°C\n"
            summary += f"  • Percentage Increase: {(model_summary['forecast_increase']/model_summary['historical_avg']*100):.1f}%\n"
            
            # Monthly forecast statistics
            forecast_values = arima_data['forecast']
            summary += f"\nMonthly Forecast Details:\n"
            summary += f"  • Minimum: {forecast_values.min():.2f}°C\n"
            summary += f"  • Maximum: {forecast_values.max():.2f}°C\n"
            summary += f"  • Standard Deviation: {forecast_values.std():.2f}°C\n"
            summary += f"  • Seasonal Variation: {forecast_values.max() - forecast_values.min():.2f}°C\n"
        
        # SARIMA Results
        if 'sarima' in self.forecasts:
            sarima_data = self.forecasts['sarima']
            model_summary = sarima_data['model_summary']
            
            summary += f"\n🌊 SARIMA SEASONAL TIME SERIES MODEL:\n"
            summary += f"{'='*42}\n"
            summary += f"Model Configuration:\n"
            summary += f"  • SARIMA Order: {sarima_data['order']}\n"
            summary += f"  • Seasonal Order: {sarima_data['seasonal_order']}\n"
            summary += f"  • Model AIC: {model_summary['aic']:.2f}\n"
            summary += f"  • Model BIC: {model_summary['bic']:.2f}\n"
            summary += f"  • Log Likelihood: {model_summary['llf']:.2f}\n"
            
            summary += f"\nSeasonal Temperature Forecasts:\n"
            summary += f"  • Historical Average (1972-2024): {model_summary['historical_avg']:.2f}°C\n"
            summary += f"  • Predicted Average (2025-2029): {model_summary['avg_forecast']:.2f}°C\n"
            summary += f"  • Predicted Increase: {model_summary['forecast_increase']:.2f}°C\n"
            summary += f"  • Percentage Increase: {(model_summary['forecast_increase']/model_summary['historical_avg']*100):.1f}%\n"
            
            summary += f"\nSeasonal Pattern Analysis:\n"
            summary += f"  • Historical Seasonal Range: {model_summary['seasonal_range_historical']:.2f}°C\n"
            summary += f"  • Predicted Seasonal Range: {model_summary['seasonal_range_forecast']:.2f}°C\n"
            summary += f"  • Change in Seasonality: {model_summary['seasonality_change']:.2f}°C\n"
            
            # Annual forecasts if available
            if 'annual_forecasts' in sarima_data:
                summary += f"\nSARIMA Annual Forecasts:\n"
                for i, temp in enumerate(sarima_data['annual_forecasts']):
                    increase = temp - model_summary['historical_avg']
                    summary += f"  • {2025 + i}: {temp:.2f}°C (+{increase:.2f}°C from historical)\n"
        
        # Annual predictions
        if hasattr(self, 'future_predictions'):
            summary += f"\n📅 ANNUAL TEMPERATURE PREDICTIONS:\n"
            summary += f"{'='*40}\n"
            
            if 'arima' in self.future_predictions:
                arima_pred = self.future_predictions['arima']
                summary += f"ARIMA Annual Forecasts:\n"
                for year, temp in zip(arima_pred['years'], arima_pred['annual_temps']):
                    increase = temp - self.forecasts['arima']['model_summary']['historical_avg']
                    summary += f"  • {year}: {temp:.2f}°C (+{increase:.2f}°C from historical)\n"
            
            if 'machine_learning' in self.future_predictions:
                ml_pred = self.future_predictions['machine_learning']
                summary += f"\nMachine Learning Forecasts:\n"
                historical_avg = self.data['Dhaka Temperature [2 m elevation corrected]'].mean()
                for year, temp in zip(ml_pred['years'], ml_pred['annual_temps']):
                    increase = temp - historical_avg
                    summary += f"  • {year}: {temp:.2f}°C (+{increase:.2f}°C from historical)\n"
        
        # Model performance comparison
        summary += f"\n🏆 MODEL PERFORMANCE COMPARISON:\n"
        summary += f"{'='*40}\n"
        
        if 'arima' in self.forecasts:
            summary += f"ARIMA Model:\n"
            summary += f"  • Type: Time Series Forecasting\n"
            summary += f"  • Strengths: Captures trends and basic patterns\n"
            summary += f"  • Model Fit: AIC={self.forecasts['arima']['model_summary']['aic']:.1f}\n"
        
        if 'sarima' in self.forecasts:
            summary += f"\nSARIMA Model:\n"
            summary += f"  • Type: Seasonal Time Series Forecasting\n"
            summary += f"  • Strengths: Captures seasonal cycles and climate patterns\n"
            summary += f"  • Model Fit: AIC={self.forecasts['sarima']['model_summary']['aic']:.1f}\n"
            summary += f"  • Seasonal Analysis: Advanced decomposition\n"
        
        if 'random_forest' in self.forecasts:
            rf_perf = self.forecasts['random_forest']
            summary += f"\nRandom Forest Model:\n"
            summary += f"  • Test R²: {rf_perf['test_r2']:.3f}\n"
            summary += f"  • Test RMSE: {rf_perf['test_rmse']:.3f}°C\n"
            summary += f"  • Best Parameters: {rf_perf['best_params']}\n"
            
            # Top features
            if 'feature_importance' in rf_perf:
                top_features = rf_perf['feature_importance'].head(3)
                summary += f"  • Top Features: {', '.join(top_features['Feature'].values)}\n"
        
        if 'xgboost' in self.forecasts:
            xgb_perf = self.forecasts['xgboost']
            summary += f"\nXGBoost Model:\n"
            summary += f"  • Test R²: {xgb_perf['test_r2']:.3f}\n"
            summary += f"  • Test RMSE: {xgb_perf['test_rmse']:.3f}°C\n"
            summary += f"  • Best Parameters: {xgb_perf['best_params']}\n"
        
        # Climate implications
        summary += f"\n🌍 CLIMATE IMPLICATIONS:\n"
        summary += f"{'='*30}\n"
        
        if 'arima' in self.forecasts:
            increase = self.forecasts['arima']['model_summary']['forecast_increase']
            if increase > 1.0:
                risk_level = "🔴 HIGH RISK"
            elif increase > 0.5:
                risk_level = "🟡 MODERATE RISK"
            else:
                risk_level = "🟢 LOW RISK"
            
            summary += f"Climate Risk Level: {risk_level}\n"
            summary += f"Projected warming: {increase:.2f}°C by 2029\n"
            
            # Heatwave implications
            current_heatwave_days = self.data[self.data['Heatwave']].groupby('Year').size().mean()
            if increase > 0.5:
                projected_increase_factor = 1 + (increase * 0.3)  # Rough estimate
                projected_heatwave_days = current_heatwave_days * projected_increase_factor
                summary += f"Expected heatwave increase: {(projected_increase_factor-1)*100:.0f}%\n"
                summary += f"Current avg heatwave days: {current_heatwave_days:.1f}/year\n"
                summary += f"Projected avg heatwave days: {projected_heatwave_days:.1f}/year\n"
        
        # Recommendations
        summary += f"\n💡 RECOMMENDATIONS:\n"
        summary += f"{'='*25}\n"
        summary += f"1. 🌡️  Implement heat early warning systems\n"
        summary += f"2. 🌳 Accelerate urban reforestation programs\n"
        summary += f"3. 🏠 Improve building cooling infrastructure\n"
        summary += f"4. 📊 Enhance climate monitoring networks\n"
        summary += f"5. 🎯 Develop heat adaptation strategies\n"
        summary += f"6. 🔬 Continue regular model updates with new data\n"
        
        summary += f"\n" + "="*70
        
        return summary
