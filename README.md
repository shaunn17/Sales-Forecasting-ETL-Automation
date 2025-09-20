# Sales Forecasting with Predictive Modeling

A comprehensive data science project for sales forecasting using multiple predictive modeling techniques, designed for portfolio demonstration.

## 🎯 Project Overview

This project demonstrates end-to-end sales forecasting capabilities including:
- Data cleaning and preprocessing
- Exploratory data analysis with visualizations
- Multiple forecasting models (SARIMA, Prophet, XGBoost)
- Business intelligence insights
- Power BI dashboard preparation

## 📁 Project Structure

```
sales_forecasting_project/
│
├── data/
│   ├── raw/                 # Original dataset
│   ├── processed/           # Cleaned and transformed datasets
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_forecast_export.ipynb
│
├── outputs/
│   ├── forecast_results.csv
│   ├── model_performance.csv
│
├── scripts/
│   ├── etl_pipeline.py
│   ├── forecasting_models.py
│
├── docs/
│   ├── report.md            # Business insights + summary
│   ├── powerbi_spec.md      # Dashboard design doc
│
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd sales_forecasting_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Generate synthetic data
python scripts/etl_pipeline.py --generate-data

# Clean and preprocess data
python scripts/etl_pipeline.py --clean-data

# Run forecasting models
python scripts/forecasting_models.py
```

### 3. Explore with Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Run notebooks in order:
# 1. 01_data_cleaning.ipynb
# 2. 02_eda.ipynb
# 3. 03_modeling.ipynb
# 4. 04_forecast_export.ipynb
```

## 📊 Key Features

### Data Processing
- Synthetic sales data generation with realistic patterns
- Comprehensive data cleaning and preprocessing
- Feature engineering (lag features, rolling averages, seasonality)

### Forecasting Models
- **Naive Baseline**: Moving averages and last period
- **SARIMA**: Seasonal ARIMA for time series forecasting
- **Prophet**: Facebook's forecasting tool with trend/seasonality
- **XGBoost**: Machine learning with engineered features

### Evaluation Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

### Business Intelligence
- Executive summary with key insights
- Power BI dashboard specifications
- Forecast scenarios (best-case, worst-case, expected)

## 📈 Outputs

The project generates:
- `outputs/forecast_results.csv`: Forecast data for Power BI
- `outputs/model_performance.csv`: Model comparison metrics
- `docs/report.md`: Business insights and recommendations
- `docs/powerbi_spec.md`: Dashboard design specifications

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **XGBoost**: Gradient boosting
- **Prophet**: Time series forecasting
- **Statsmodels**: Statistical models (SARIMA)
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive notebooks
- **Power BI**: Business intelligence dashboard

## 📝 Notebook Walkthrough

1. **01_data_cleaning.ipynb**: Data loading, cleaning, and preprocessing
2. **02_eda.ipynb**: Exploratory data analysis and visualizations
3. **03_modeling.ipynb**: Model training, evaluation, and comparison
4. **04_forecast_export.ipynb**: Forecast generation and export for BI

## 🎯 Business Applications

This project demonstrates skills relevant for:
- Data Science roles
- Business Intelligence positions
- Forecasting and planning teams
- Analytics consulting

## 📊 Sample Insights

- Sales trend analysis and seasonality detection
- Regional performance comparisons
- Product category forecasting
- Anomaly detection and alerting
- What-if scenario analysis

## 🤝 Contributing

This is a portfolio project. Feel free to fork and modify for your own use.

## 📄 License

MIT License - see LICENSE file for details.
