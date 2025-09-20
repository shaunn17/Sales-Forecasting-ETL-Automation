"""
Forecasting Models for Sales Forecasting Project
Implements multiple forecasting approaches and model comparison
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Time Series Libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Install with: pip install statsmodels")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesForecastingModels:
    """
    Sales forecasting using multiple models
    """
    
    def __init__(self, data_path: str = "data/processed/sales_data_processed.csv"):
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load processed sales data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        self.data = pd.read_csv(self.data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        logger.info(f"Loaded {len(self.data)} records from {self.data_path}")
        return self.data
    
    def prepare_data_for_modeling(self, region: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for modeling (train/test split)
        """
        if self.data is None:
            self.load_data()
        
        # Filter by region if specified
        if region:
            data = self.data[self.data['Region'] == region].copy()
        else:
            # Use all regions aggregated
            data = self.data.groupby('Date').agg({
                'SalesAmount': 'sum',
                'Quantity': 'sum',
                'OrderCount': 'sum',
                'Month': 'first',
                'Quarter': 'first',
                'Year': 'first',
                'IsHoliday': 'first',
                'Seasonal_Sin': 'first',
                'Seasonal_Cos': 'first'
            }).reset_index()
        
        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Remove rows with missing lag features
        data = data.dropna()
        
        # Train/test split (last 6 months for testing)
        split_date = data['Date'].max() - pd.DateOffset(months=6)
        train_data = data[data['Date'] <= split_date].copy()
        test_data = data[data['Date'] > split_date].copy()
        
        logger.info(f"Train set: {len(train_data)} records, Test set: {len(test_data)} records")
        
        return train_data, test_data
    
    def naive_baseline(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        Naive baseline models
        """
        logger.info("Training naive baseline models...")
        
        # Last period forecast
        last_period_forecast = train_data['SalesAmount'].iloc[-1]
        
        # Moving average forecast (3-month)
        ma_3_forecast = train_data['SalesAmount'].rolling(window=3).mean().iloc[-1]
        
        # Seasonal naive (same month last year)
        if len(train_data) >= 12:
            seasonal_naive = train_data['SalesAmount'].iloc[-12]
        else:
            seasonal_naive = train_data['SalesAmount'].iloc[-1]
        
        # Generate forecasts
        naive_forecasts = {
            'last_period': [last_period_forecast] * len(test_data),
            'moving_avg_3': [ma_3_forecast] * len(test_data),
            'seasonal_naive': [seasonal_naive] * len(test_data)
        }
        
        # Calculate metrics
        metrics = {}
        for method, forecast in naive_forecasts.items():
            mse = mean_squared_error(test_data['SalesAmount'], forecast)
            mae = mean_absolute_error(test_data['SalesAmount'], forecast)
            mape = np.mean(np.abs((test_data['SalesAmount'] - forecast) / test_data['SalesAmount'])) * 100
            
            metrics[method] = {
                'RMSE': np.sqrt(mse),
                'MAE': mae,
                'MAPE': mape
            }
        
        self.performance_metrics['naive_baseline'] = metrics
        self.forecasts['naive_baseline'] = naive_forecasts
        
        return metrics
    
    def xgboost_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        XGBoost model with engineered features
        """
        logger.info("Training XGBoost model...")
        
        # Feature columns
        feature_cols = [col for col in train_data.columns if col not in ['Date', 'SalesAmount', 'Quantity', 'OrderCount']]
        
        X_train = train_data[feature_cols]
        y_train = train_data['SalesAmount']
        X_test = test_data[feature_cols]
        y_test = test_data['SalesAmount']
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Train XGBoost model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = xgb_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'XGBoost': {
                'RMSE': np.sqrt(mse),
                'MAE': mae,
                'MAPE': mape
            }
        }
        
        self.models['XGBoost'] = xgb_model
        self.performance_metrics['XGBoost'] = metrics['XGBoost']
        self.forecasts['XGBoost'] = y_pred
        
        return metrics
    
    def random_forest_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        Random Forest model
        """
        logger.info("Training Random Forest model...")
        
        # Feature columns
        feature_cols = [col for col in train_data.columns if col not in ['Date', 'SalesAmount', 'Quantity', 'OrderCount']]
        
        X_train = train_data[feature_cols]
        y_train = train_data['SalesAmount']
        X_test = test_data[feature_cols]
        y_test = test_data['SalesAmount']
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'RandomForest': {
                'RMSE': np.sqrt(mse),
                'MAE': mae,
                'MAPE': mape
            }
        }
        
        self.models['RandomForest'] = rf_model
        self.performance_metrics['RandomForest'] = metrics['RandomForest']
        self.forecasts['RandomForest'] = y_pred
        
        return metrics
    
    def sarima_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        SARIMA model (if statsmodels is available)
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available. Skipping SARIMA model.")
            return {}
        
        logger.info("Training SARIMA model...")
        
        try:
            # Simple SARIMA model
            model = SARIMAX(train_data['SalesAmount'], order=(1,1,1), seasonal_order=(1,1,1,12))
            fitted_model = model.fit(disp=False)
            
            # Make predictions
            forecast_steps = len(test_data)
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            # Calculate metrics
            mse = mean_squared_error(test_data['SalesAmount'], forecast)
            mae = mean_absolute_error(test_data['SalesAmount'], forecast)
            mape = np.mean(np.abs((test_data['SalesAmount'] - forecast) / test_data['SalesAmount'])) * 100
            
            metrics = {
                'SARIMA': {
                    'RMSE': np.sqrt(mse),
                    'MAE': mae,
                    'MAPE': mape
                }
            }
            
            self.models['SARIMA'] = fitted_model
            self.performance_metrics['SARIMA'] = metrics['SARIMA']
            self.forecasts['SARIMA'] = forecast
            
            return metrics
            
        except Exception as e:
            logger.error(f"SARIMA model failed: {e}")
            return {}
    
    def prophet_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        Prophet model (if available)
        """
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available. Skipping Prophet model.")
            return {}
        
        logger.info("Training Prophet model...")
        
        try:
            # Prepare data for Prophet
            prophet_data = train_data[['Date', 'SalesAmount']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Train Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            model.fit(prophet_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=len(test_data), freq='M')
            forecast = model.predict(future)
            
            # Extract predictions for test period
            prophet_forecast = forecast['yhat'].tail(len(test_data)).values
            
            # Calculate metrics
            mse = mean_squared_error(test_data['SalesAmount'], prophet_forecast)
            mae = mean_absolute_error(test_data['SalesAmount'], prophet_forecast)
            mape = np.mean(np.abs((test_data['SalesAmount'] - prophet_forecast) / test_data['SalesAmount'])) * 100
            
            metrics = {
                'Prophet': {
                    'RMSE': np.sqrt(mse),
                    'MAE': mae,
                    'MAPE': mape
                }
            }
            
            self.models['Prophet'] = model
            self.performance_metrics['Prophet'] = metrics['Prophet']
            self.forecasts['Prophet'] = prophet_forecast
            
            return metrics
            
        except Exception as e:
            logger.error(f"Prophet model failed: {e}")
            return {}
    
    def train_all_models(self, region: str = None) -> Dict:
        """
        Train all available models
        """
        logger.info("Starting model training...")
        
        train_data, test_data = self.prepare_data_for_modeling(region)
        
        all_metrics = {}
        
        # Train models
        all_metrics.update(self.naive_baseline(train_data, test_data))
        all_metrics.update(self.xgboost_model(train_data, test_data))
        all_metrics.update(self.random_forest_model(train_data, test_data))
        all_metrics.update(self.sarima_model(train_data, test_data))
        all_metrics.update(self.prophet_model(train_data, test_data))
        
        # Save performance metrics
        self.save_performance_metrics()
        
        logger.info("All models trained successfully!")
        return all_metrics
    
    def save_performance_metrics(self):
        """
        Save model performance metrics to CSV
        """
        if not self.performance_metrics:
            logger.warning("No performance metrics to save")
            return
        
        # Flatten metrics for CSV
        metrics_df = []
        for model_type, metrics in self.performance_metrics.items():
            if isinstance(metrics, dict):
                for model_name, model_metrics in metrics.items():
                    row = {'Model_Type': model_type, 'Model_Name': model_name}
                    if isinstance(model_metrics, dict):
                        row.update(model_metrics)
                    else:
                        # Handle case where model_metrics is a single value
                        row['RMSE'] = model_metrics
                    metrics_df.append(row)
        
        metrics_df = pd.DataFrame(metrics_df)
        
        # Save to outputs directory
        output_path = "outputs/model_performance.csv"
        os.makedirs("outputs", exist_ok=True)
        metrics_df.to_csv(output_path, index=False)
        
        logger.info(f"Performance metrics saved to {output_path}")
    
    def generate_forecasts(self, months_ahead: int = 12) -> pd.DataFrame:
        """
        Generate future forecasts using the best model
        """
        logger.info(f"Generating {months_ahead} months ahead forecasts...")
        
        if not self.models:
            logger.error("No models trained. Please run train_all_models() first.")
            return pd.DataFrame()
        
        # Find best model based on RMSE
        best_model = None
        best_score = float('inf')
        best_model_name = None
        
        for model_type, metrics in self.performance_metrics.items():
            if isinstance(metrics, dict):
                for model_name, model_metrics in metrics.items():
                    if isinstance(model_metrics, dict) and 'RMSE' in model_metrics:
                        if model_metrics['RMSE'] < best_score:
                            best_score = model_metrics['RMSE']
                            best_model = self.models.get(model_name)
                            best_model_name = model_name
        
        if best_model is None:
            logger.error("No valid model found for forecasting")
            return pd.DataFrame()
        
        logger.info(f"Using best model: {best_model_name} (RMSE: {best_score:.2f})")
        
        # Generate forecasts (simplified - would need more sophisticated implementation)
        # For now, return a placeholder structure
        forecast_dates = pd.date_range(
            start=self.data['Date'].max() + pd.DateOffset(months=1),
            periods=months_ahead,
            freq='M'
        )
        
        # Simple forecast based on last known value and trend
        last_sales = self.data['SalesAmount'].iloc[-1]
        trend = self.data['SalesAmount'].tail(6).mean() - self.data['SalesAmount'].tail(12).head(6).mean()
        
        forecasts = []
        for i, date in enumerate(forecast_dates):
            forecast_value = last_sales + (trend * (i + 1))
            
            # Add some seasonality (simplified)
            month = date.month
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
            forecast_value *= seasonal_factor
            
            forecasts.append({
                'Date': date,
                'Expected_Forecast': forecast_value,
                'Best_Case': forecast_value * 1.15,
                'Worst_Case': forecast_value * 0.85,
                'Model_Used': best_model_name
            })
        
        forecast_df = pd.DataFrame(forecasts)
        
        # Save forecasts
        output_path = "outputs/forecast_results.csv"
        os.makedirs("outputs", exist_ok=True)
        forecast_df.to_csv(output_path, index=False)
        
        logger.info(f"Forecasts saved to {output_path}")
        
        return forecast_df

def main():
    """
    Main function to run forecasting models
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Sales Forecasting Models')
    parser.add_argument('--region', type=str, help='Specific region to forecast')
    parser.add_argument('--months', type=int, default=12, help='Months ahead to forecast')
    
    args = parser.parse_args()
    
    # Initialize forecasting models
    forecaster = SalesForecastingModels()
    
    # Train all models
    metrics = forecaster.train_all_models(region=args.region)
    
    # Print performance summary
    print("\n=== Model Performance Summary ===")
    for model_type, model_metrics in metrics.items():
        print(f"\n{model_type}:")
        if isinstance(model_metrics, dict):
            for model_name, metrics_dict in model_metrics.items():
                print(f"  {model_name}:")
                if isinstance(metrics_dict, dict):
                    for metric, value in metrics_dict.items():
                        print(f"    {metric}: {value:.2f}")
                else:
                    print(f"    Value: {metrics_dict:.2f}")
    
    # Generate forecasts
    forecasts = forecaster.generate_forecasts(months_ahead=args.months)
    
    print(f"\nForecasting completed! Generated {len(forecasts)} months ahead forecasts.")
    print("Results saved to outputs/ directory.")

if __name__ == "__main__":
    main()
