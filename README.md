# BIXI Station Hourly Demand Prediction
**Live demo:** https://bixi-demand-dashboard.streamlit.app/

---
![1](https://github.com/user-attachments/assets/be920a54-8920-4684-b21a-f190809852b1)

---
![2](https://github.com/user-attachments/assets/fd0ca475-885d-44b0-9c9d-f8534ac9ef9e)

---
## Project Overview
This project builds an **end-to-end machine learning pipeline** to predict **hourly bike-sharing demand** for **BIXI stations in Montreal**. Using **historical BIXI trip data** and **Montreal weather data**, the pipeline performs data cleaning and feature engineering on **temporal and weather features**, trains a **LightGBM regression model** with **Bayesian hyperparameter optimization**, and groups stations into demand tiers using **K-Means clustering**. The Streamlit app integrates a **16-day weather forecast** from the **Open-Meteo API** and visualizes station clusters with a **PyDeck heatmap** to support **station-level operational planning**.

---
## Repository Structure

```
├── data/
│   ├── .gitattributes
│   └── model_df.zip           # Feature-engineered dataset: output of the first notebook and input to the last two notebooks
├── notebooks/
│   ├── data_cleaning_eda_feature_engineering.ipynb  
│   ├── model_clustering.ipynb # Station clustering analysis
│   └── model_lightgbm.ipynb   # Prediction model training & evaluation
├── app.py                     # Streamlit dashboard app; requires the five files below as inputs
├── model_lightgbm.txt         # Trained LightGBM model
├── meta_lightgbm.pkl          # Model metadata & feature lookups
├── station_clusters.csv       # Station cluster assignments
├── requirements.txt           # Python dependencies
└── runtime.txt                # Python runtime version
```

---
## Workflow

### 1. Data Cleaning, EDA & Feature Engineering
**Notebook:** `data_cleaning_eda_feature_engineering.ipynb`

- **Data Sources:** [BIXI Montreal Open Data](https://bixi.com/en/open-data/) and [Montreal Weather Open Data](https://montreal.weatherstats.ca/download.html)
- **Cleaning Steps:**
  - Convert timestamps to datetime and filter the data to 2024 and May/Oct 2025
  - Fill missing values using column-wise means and linear interpolation
  - Remove invalid trips (unfillable missing values; outlier durations)
  - Filter to the top 400 stations by 2024 trip volume
- **Exploratory Analysis:**
  - Demand patterns by hour, day of week, and month
  - Holiday vs. non-holiday demand comparison
  - Distribution and correlation analyses of weather variables
- **Features engineered:**

| Feature | Description |
|---------|-------------|
| `station_hour_demand_24` | Mean 2024 demand for station × hour |
| `station_dow_demand_24` | Mean 2024 demand for station × day-of-week |
| `station_month_demand_24` | Mean 2024 demand for station × month |
| `hour`, `dow`, `month` | Temporal indicators |
| `is_holiday` | Quebec/Montreal public holiday flag |
| `temperature`, `feels_like` | Hourly temperature metrics |
| `wind_speed` | Wind speed in km/h |
| `bad_weather` | Binary flag (humidity > 85% and visibility < 10km) |

**Target Variable:** `total_demand` (sum of departures and returns per station per hour)

---
### 2. Station Clustering
**Notebook:** `model_clustering.ipynb`

- **Algorithm:** K-Means (k=3)
- **Clustering Feature:** Mean hourly demand per station (2024)
- **Output Clusters:**
  - **Low demand:** 247 stations (~7.4 avg trips/hour)
  - **Medium demand:** 120 stations (~12.2 avg trips/hour)
  - **High demand:** 33 stations (~19.2 avg trips/hour)
- **Validation:** Silhouette score indicates moderate-to-strong cluster separation

---
### 3. Model Training & Evaluation
**Notebook:** `model_lightgbm.ipynb`

- **Algorithm:** LightGBM (Gradient Boosting Decision Trees)
- **Data Split (chronological):**
  - Training: 2024 data (83%) — build 2024-based baseline features for forecasting
  - Validation: May 2025 (9%)
  - Test: Oct 2025 (8%) — May and Oct show moderate demand and are more representative months for validation and testing
- **Hyperparameter Tuning:** Bayesian optimization via Optuna (40 trials)
- **Evaluation Metrics:**

| Dataset | R² | RMSE | MAE |
|---------|-----|------|-----|
| Train (2024) | 0.72 | 5.08 | 3.21 |
| Validation (May 2025) | 0.64 | 5.75 | 3.73 |
| Test (Oct 2025) | 0.63 | 5.85 | 3.82 |

- **Model Interpretation:** SHAP analysis reveals top predictors are `station_hour_demand_24`, `station_month_demand_24`, and `temperature`

---
### 4. Streamlit Application
**File:** `app.py`

The dashboard provides three views:

1. **16-Day Demand Forecast**
   - Uses real-time weather data from Open-Meteo API
   - Single time-point prediction or full-day hourly forecast
   - Interactive charts with weather overlay

2. **Custom Input Forecast**
   - Manual weather parameter entry
   - Predictions for any future date

3. **Station Clusters Visualization**
   - Interactive PyDeck heatmap
   - Filter by cluster (low/medium/high)
   - Station-level tooltips with demand statistics

## Technical Stack

| Category | Tools |
|----------|-------|
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly, PyDeck |
| **Machine Learning** | LightGBM, scikit-learn, Optuna |
| **Explainability** | SHAP |
| **Web Application** | Streamlit |
| **Weather API** | Open-Meteo |

## Data Sources

- **BIXI Trip Data:** 
- **Weather Data:** [Open-Meteo API](https://open-meteo.com/)

## Notes

- Demand is defined as the **sum of departures and returns** at each station per hour
- The model uses 2024 historical patterns as baseline features for future predictions
- Only the top 400 stations by trip volume are included in the analysis
- Weather forecasts are limited to 16 days ahead (API constraint)
