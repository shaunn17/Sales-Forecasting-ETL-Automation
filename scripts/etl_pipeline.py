"""
ETL Pipeline for Sales Forecasting Project
Downloads and processes UCI Online Retail dataset
"""

import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesETLPipeline:
    """
    ETL Pipeline for processing UCI Online Retail dataset
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def download_uci_data(self) -> str:
        """
        Download UCI Online Retail dataset
        Returns path to downloaded file
        """
        logger.info("Downloading UCI Online Retail dataset...")
        
        # UCI dataset URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        
        file_path = os.path.join(self.raw_dir, "online_retail.xlsx")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    
            logger.info(f"Dataset downloaded successfully to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw UCI dataset
        """
        if file_path is None:
            file_path = os.path.join(self.raw_dir, "online_retail.xlsx")
            
        if not os.path.exists(file_path):
            file_path = self.download_uci_data()
            
        logger.info("Loading raw dataset...")
        df = pd.read_excel(file_path)
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataset
        """
        logger.info("Starting data cleaning...")
        
        # Create a copy
        df_clean = df.copy()
        
        # Remove rows with missing InvoiceNo or CustomerID
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['InvoiceNo', 'CustomerID'])
        logger.info(f"Removed {initial_count - len(df_clean)} rows with missing InvoiceNo or CustomerID")
        
        # Remove rows with negative quantities or prices
        df_clean = df_clean[df_clean['Quantity'] > 0]
        df_clean = df_clean[df_clean['UnitPrice'] > 0]
        logger.info(f"Removed rows with negative quantities or prices")
        
        # Remove cancelled orders (InvoiceNo starting with 'C')
        df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
        logger.info(f"Removed cancelled orders")
        
        # Convert InvoiceDate to datetime
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        
        # Remove outliers (very high quantities or prices)
        q1_qty, q3_qty = df_clean['Quantity'].quantile([0.25, 0.75])
        iqr_qty = q3_qty - q1_qty
        df_clean = df_clean[df_clean['Quantity'] <= q3_qty + 1.5 * iqr_qty]
        
        q1_price, q3_price = df_clean['UnitPrice'].quantile([0.25, 0.75])
        iqr_price = q3_price - q1_price
        df_clean = df_clean[df_clean['UnitPrice'] <= q3_price + 1.5 * iqr_price]
        
        logger.info(f"Removed outliers. Final dataset: {len(df_clean)} records")
        
        return df_clean
    
    def aggregate_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sales data by month, region, and product category
        """
        logger.info("Aggregating sales data...")
        
        # Create month-year column
        df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
        
        # Calculate total sales amount
        df['TotalSales'] = df['Quantity'] * df['UnitPrice']
        
        # Create product categories (simplified)
        df['ProductCategory'] = df['Description'].str[:20]  # First 20 chars as category
        
        # Aggregate by month and country (as region)
        monthly_sales = df.groupby(['YearMonth', 'Country']).agg({
            'TotalSales': 'sum',
            'Quantity': 'sum',
            'InvoiceNo': 'nunique'  # Count of unique invoices
        }).reset_index()
        
        monthly_sales.columns = ['YearMonth', 'Region', 'SalesAmount', 'Quantity', 'OrderCount']
        
        # Convert YearMonth back to datetime
        monthly_sales['Date'] = monthly_sales['YearMonth'].dt.to_timestamp()
        
        # Sort by date
        monthly_sales = monthly_sales.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Aggregated to {len(monthly_sales)} monthly records")
        
        return monthly_sales
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for forecasting
        """
        logger.info("Creating engineered features...")
        
        df_features = df.copy()
        
        # Lag features (1, 3, 6 months)
        for lag in [1, 3, 6]:
            df_features[f'SalesAmount_lag_{lag}'] = df_features.groupby('Region')['SalesAmount'].shift(lag)
            df_features[f'Quantity_lag_{lag}'] = df_features.groupby('Region')['Quantity'].shift(lag)
        
        # Rolling averages (3-month, 6-month)
        for window in [3, 6]:
            df_features[f'SalesAmount_ma_{window}'] = df_features.groupby('Region')['SalesAmount'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df_features[f'Quantity_ma_{window}'] = df_features.groupby('Region')['Quantity'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
        
        # Seasonal features
        df_features['Month'] = df_features['Date'].dt.month
        df_features['Quarter'] = df_features['Date'].dt.quarter
        df_features['Year'] = df_features['Date'].dt.year
        
        # Holiday flags (simplified - December and January)
        df_features['IsHoliday'] = df_features['Month'].isin([12, 1])
        
        # Growth rate (month-over-month)
        df_features['SalesGrowth'] = df_features.groupby('Region')['SalesAmount'].pct_change()
        
        # Seasonal decomposition features (simplified)
        df_features['Seasonal_Sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
        df_features['Seasonal_Cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
        
        logger.info("Feature engineering completed")
        
        return df_features
    
    def process_full_pipeline(self) -> pd.DataFrame:
        """
        Run the complete ETL pipeline
        """
        logger.info("Starting full ETL pipeline...")
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Clean data
        cleaned_data = self.clean_data(raw_data)
        
        # Aggregate sales
        aggregated_data = self.aggregate_sales(cleaned_data)
        
        # Create features
        final_data = self.create_features(aggregated_data)
        
        # Save processed data
        output_path = os.path.join(self.processed_dir, "sales_data_processed.csv")
        final_data.to_csv(output_path, index=False)
        
        logger.info(f"Pipeline completed. Processed data saved to {output_path}")
        
        return final_data

def main():
    """
    Main function to run ETL pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='ETL Pipeline for Sales Forecasting')
    parser.add_argument('--download-only', action='store_true', help='Only download the dataset')
    parser.add_argument('--clean-only', action='store_true', help='Only clean existing data')
    
    args = parser.parse_args()
    
    pipeline = SalesETLPipeline()
    
    if args.download_only:
        pipeline.download_uci_data()
    elif args.clean_only:
        raw_data = pipeline.load_raw_data()
        processed_data = pipeline.process_full_pipeline()
    else:
        processed_data = pipeline.process_full_pipeline()
        
    print("ETL Pipeline completed successfully!")

if __name__ == "__main__":
    main()
