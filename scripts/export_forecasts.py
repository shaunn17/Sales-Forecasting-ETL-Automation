"""
Export Forecasts for Power BI
Generates forecast results in Power BI-ready format
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_forecast_export():
    """
    Generate forecast export for Power BI
    """
    logger.info("Starting forecast export...")
    
    # Load processed data
    data_path = "data/processed/sales_data_processed.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Aggregate monthly data
    monthly_data = data.groupby('Date').agg({
        'SalesAmount': 'sum',
        'Quantity': 'sum',
        'OrderCount': 'sum'
    }).reset_index()
    
    # Generate forecast dates (next 12 months)
    last_date = monthly_data['Date'].max()
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=12,
        freq='M'
    )
    
    # Simple forecasting based on trend and seasonality
    def generate_forecasts(historical_data, forecast_dates):
        """Generate forecasts using trend and seasonality"""
        
        # Calculate trend (last 6 months vs previous 6 months)
        recent_avg = historical_data['SalesAmount'].tail(6).mean()
        previous_avg = historical_data['SalesAmount'].tail(12).head(6).mean()
        trend_factor = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
        
        # Calculate seasonal factors
        monthly_factors = {}
        for month in range(1, 13):
            month_data = historical_data[historical_data['Date'].dt.month == month]
            if len(month_data) > 0:
                monthly_factors[month] = month_data['SalesAmount'].mean() / historical_data['SalesAmount'].mean()
            else:
                monthly_factors[month] = 1.0
        
        # Base forecast (last value with trend)
        base_sales = historical_data['SalesAmount'].iloc[-1]
        
        forecasts = []
        for i, date in enumerate(forecast_dates):
            month = date.month
            
            # Apply trend
            trend_adjusted = base_sales * (1 + trend_factor * (i + 1) / 12)
            
            # Apply seasonality
            seasonal_factor = monthly_factors.get(month, 1.0)
            expected_forecast = trend_adjusted * seasonal_factor
            
            # Create scenarios
            best_case = expected_forecast * 1.15  # 15% optimistic
            worst_case = expected_forecast * 0.85  # 15% pessimistic
            
            forecasts.append({
                'Date': date,
                'Expected_Forecast': expected_forecast,
                'Best_Case': best_case,
                'Worst_Case': worst_case,
                'Model_Used': 'Trend_Seasonal',
                'Month': month,
                'Quarter': (month - 1) // 3 + 1,
                'Year': date.year,
                'Is_Forecast': True
            })
        
        return pd.DataFrame(forecasts)
    
    # Generate forecasts
    forecast_results = generate_forecasts(monthly_data, forecast_dates)
    
    # Prepare historical data for export
    historical_export = monthly_data.copy()
    historical_export['Expected_Forecast'] = historical_export['SalesAmount']
    historical_export['Best_Case'] = historical_export['SalesAmount']
    historical_export['Worst_Case'] = historical_export['SalesAmount']
    historical_export['Model_Used'] = 'Actual'
    historical_export['Month'] = historical_export['Date'].dt.month
    historical_export['Quarter'] = historical_export['Date'].dt.quarter
    historical_export['Year'] = historical_export['Date'].dt.year
    historical_export['Is_Forecast'] = False
    
    # Combine datasets
    powerbi_data = pd.concat([
        historical_export[['Date', 'Expected_Forecast', 'Best_Case', 'Worst_Case', 
                          'Model_Used', 'Month', 'Quarter', 'Year', 'Is_Forecast']],
        forecast_results
    ], ignore_index=True)
    
    # Add additional metrics for Power BI
    powerbi_data['YoY_Growth'] = powerbi_data.groupby(powerbi_data['Date'].dt.month)['Expected_Forecast'].pct_change(12) * 100
    powerbi_data['MoM_Growth'] = powerbi_data['Expected_Forecast'].pct_change() * 100
    powerbi_data['Forecast_Confidence'] = np.where(
        powerbi_data['Is_Forecast'], 
        100 - (powerbi_data['Month'] - 1) * 5,  # Decreasing confidence over time
        100
    )
    
    # Save to outputs directory
    output_path = "outputs/forecast_results.csv"
    os.makedirs("outputs", exist_ok=True)
    powerbi_data.to_csv(output_path, index=False)
    
    logger.info(f"Forecast data exported to: {output_path}")
    logger.info(f"Total records: {len(powerbi_data)}")
    logger.info(f"Historical records: {len(historical_export)}")
    logger.info(f"Forecast records: {len(forecast_results)}")
    
    # Print summary
    total_expected = forecast_results['Expected_Forecast'].sum()
    total_best = forecast_results['Best_Case'].sum()
    total_worst = forecast_results['Worst_Case'].sum()
    
    print(f"\n📊 Forecast Summary (Next 12 Months):")
    print(f"Expected Total Sales: ${total_expected:,.2f}")
    print(f"Best Case Total Sales: ${total_best:,.2f}")
    print(f"Worst Case Total Sales: ${total_worst:,.2f}")
    print(f"Average Monthly Sales: ${total_expected/12:,.2f}")
    
    return powerbi_data

def main():
    """Main function"""
    try:
        forecast_data = generate_forecast_export()
        print("\n✅ Forecast export completed successfully!")
        print("✅ Power BI data ready for dashboard creation!")
        
    except Exception as e:
        logger.error(f"Error in forecast export: {e}")
        raise

if __name__ == "__main__":
    main()
