"""
HOW TO RUN THIS APP:
1. Activate your conda environment
2. Navigate to the app directory
3. Run the Streamlit app: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import datetime as dt
import pydeck as pdk
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =================================================
# GLOBAL PAGE CONFIG
# =================================================
st.set_page_config(page_title="BIXI Demand Dashboard", layout="wide")
st.title("🚲 BIXI Station Hourly Demand Dashboard")


# =================================================
# MODEL LOADING
# Loads the LightGBM model and metadata
# =================================================
@st.cache_resource
def load_model_assets():
    booster = lgb.Booster(model_file="model_lightgbm.txt")
    with open("meta_lightgbm.pkl", "rb") as f:
        meta = pickle.load(f)
    return booster, meta


# =================================================
# WEATHER API FUNCTIONS
# Fetches 16-day weather forecast from Open-Meteo API for Montreal
# =================================================
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_weather_data():
    # Montreal coordinates
    LAT = 45.5017
    LON = -73.5673

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": (
            "temperature_2m,"
            "apparent_temperature,"
            "wind_speed_10m,"
            "relative_humidity_2m,"
            "visibility"
        ),
        "forecast_days": 16,
        "timezone": "America/Toronto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
    }

    try:
        # Make API request with timeout
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Parse hourly data into a DataFrame
        hourly = data["hourly"]
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(hourly["time"]),
                "temperature": hourly["temperature_2m"],
                "feels_like": hourly["apparent_temperature"],
                "wind_speed": hourly["wind_speed_10m"],
                "relative_humidity": hourly["relative_humidity_2m"],
                "visibility": hourly["visibility"],
            }
        )
        return df, None
    except requests.RequestException as e:
        return None, f"Request failed: {e}"
    except KeyError as e:
        return None, f"Unexpected response format, missing key: {e}"


# Look up weather values for a specific datetime from the weather DataFrame
def get_weather_for_datetime(weather_df, target_datetime):
    # Find the row matching the target datetime
    mask = weather_df["datetime"] == target_datetime
    if mask.any():
        row = weather_df.loc[mask].iloc[0]
        bad_weather = 1 if (row["relative_humidity"] > 85 and row["visibility"] < 10000) else 0
        
        return {
            "temperature": row["temperature"],
            "feels_like": row["feels_like"],
            "wind_speed": row["wind_speed"],
            "bad_weather": bad_weather,
            "humidity": row["relative_humidity"],
            "visibility": row["visibility"],
        }
    return None


# Get weather data for all 24 hours of a specific date
def get_weather_for_day(weather_df, target_date):
    # Filter weather data for the target date
    mask = weather_df["datetime"].dt.date == target_date
    day_df = weather_df[mask].copy()
    
    if day_df.empty:
        return None
    
    # Extract hour and compute bad_weather flag
    day_df["hour"] = day_df["datetime"].dt.hour
    day_df["bad_weather"] = (
        (day_df["relative_humidity"] > 85) &
        (day_df["visibility"] < 10000)
    ).astype(int)
    return day_df


# =================================================
# FEATURE BUILDING FUNCTIONS
# =================================================
def build_features(
    station,
    date,
    hour,
    is_holiday,
    temperature,
    feels_like,
    wind_speed,
    bad_weather,
    meta
):
    # Compute day of week (1=Monday, 7=Sunday) and month
    dow = date.weekday() + 1
    month = date.month

    # Get station-level demand lookup tables from metadata
    station_hour_demand_24_lookup = meta["station_hour_demand_24"]
    station_dow_demand_24_lookup = meta["station_dow_demand_24"]
    station_month_demand_24_lookup = meta["station_month_demand_24"]
    
    # Global fallback values if station-specific data not available
    global_hour_demand_24 = meta["global_hour_demand_24"]
    global_dow_demand_24 = meta["global_dow_demand_24"]
    global_month_demand_24 = meta["global_month_demand_24"]

    # Look up station-specific demand statistics, fallback to global if not found
    key_hour = (station, hour)
    key_dow = (station, dow)
    key_month = (station, month)
    station_hour_demand_24 = station_hour_demand_24_lookup.get(key_hour, global_hour_demand_24)
    station_dow_demand_24 = station_dow_demand_24_lookup.get(key_dow, global_dow_demand_24)
    station_month_demand_24 = station_month_demand_24_lookup.get(key_month, global_month_demand_24)

    return {
        "station": station,
        "station_hour_demand_24": station_hour_demand_24,
        "station_dow_demand_24": station_dow_demand_24,
        "station_month_demand_24": station_month_demand_24,
        "hour": hour,
        "dow": dow,
        "month": month,
        "is_holiday": is_holiday,
        "temperature": temperature,
        "feels_like": feels_like,
        "wind_speed": wind_speed,
        "bad_weather": bad_weather,
    }


# =================================================
# PAGE 1: 16-Day Demand Forecast (API Weather)
# Contains two tabs:
#   - Tab 1: Single Time Point Prediction
#   - Tab 2: Prediction for a Day
# =================================================
def render_page1():
    # Load model and metadata
    booster, meta = load_model_assets()

    st.header("16-Day Demand Forecast")
    st.write(
        "Select a station, date, and hour to get a demand prediction using "
        "**real-time weather data** from the Open-Meteo API."
    )

    # -----------------------------------------------
    # Fetch weather data from API
    # -----------------------------------------------
    weather_df, error = fetch_weather_data()
    
    if error:
        st.error(f"Failed to fetch weather data: {error}")
        st.info("Please try again later or use the 'Demand Forecast with Custom Inputs' page.")
        return
    
    if weather_df is None or weather_df.empty:
        st.error("No weather data available.")
        return

    # Get available date range from weather data
    min_datetime = weather_df["datetime"].min()
    max_datetime = weather_df["datetime"].max()
    min_date = min_datetime.date()
    max_date = max_datetime.date()

    # Get station list from metadata
    station_list = meta["station"]

    # Create two tabs
    tab1, tab2 = st.tabs(["Single Time Point Prediction", "Prediction for a Day"])

    # ===============================================
    # TAB 1: Single Time Point Prediction
    # ===============================================
    with tab1:
        st.subheader("Predict demand for a specific hour")
        
        # Station selection dropdown
        station_input_tab1 = st.selectbox(
            "Station", station_list, key="page1_tab1_station"
        )

        # Date and Holiday on the same row
        col_date, col_holiday = st.columns(2)
        with col_date:
            date_input_tab1 = st.date_input(
                "Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="page1_tab1_date",
                help=f"Weather data available from {min_date} to {max_date}"
            )
        with col_holiday:
            is_holiday_input_tab1 = st.selectbox(
                "Holiday (0=No, 1=Yes)", options=[0, 1], index=0, key="page1_tab1_hol"
            )

        # Hour slider
        hour_input_tab1 = st.slider("Hour of day", 0, 23, 12, key="page1_tab1_hour")

        # Construct target datetime for weather lookup
        target_datetime = pd.Timestamp(
            year=date_input_tab1.year,
            month=date_input_tab1.month,
            day=date_input_tab1.day,
            hour=hour_input_tab1
        )

        # Look up weather for the selected datetime
        weather_values = get_weather_for_datetime(weather_df, target_datetime)

        if weather_values is None:
            st.warning(f"No weather data available for {target_datetime}. Please select a different date/hour.")
        else:
            # Display the fetched weather values
            st.markdown("### Weather Conditions (from API)")
            w_col1, w_col2, w_col3, w_col4 = st.columns(4)
            with w_col1:
                st.metric("Temperature", f"{weather_values['temperature']:.1f} °C")
            with w_col2:
                st.metric("Feels Like", f"{weather_values['feels_like']:.1f} °C")
            with w_col3:
                st.metric("Wind Speed", f"{weather_values['wind_speed']:.1f} km/h")
            with w_col4:
                bad_weather_label = "Yes" if weather_values['bad_weather'] == 1 else "No"
                st.metric("Bad Weather", bad_weather_label)

            # Additional weather info in expander
            with st.expander("Additional weather details"):
                st.write(f"**Relative Humidity:** {weather_values['humidity']:.1f}%")
                st.write(f"**Visibility:** {weather_values['visibility']:.0f} m")
                st.write("*Bad weather is defined as humidity > 85% and visibility < 10,000 m*")

            # Build features for prediction
            feature_values = build_features(
                station=station_input_tab1,
                date=date_input_tab1,
                hour=hour_input_tab1,
                is_holiday=int(is_holiday_input_tab1),
                temperature=weather_values["temperature"],
                feels_like=weather_values["feels_like"],
                wind_speed=weather_values["wind_speed"],
                bad_weather=weather_values["bad_weather"],
                meta=meta,
            )

            all_features = meta["all_features"]
            categorical_features = meta["categorical_features"]

            # Predict button
            if st.button("Predict demand", key="page1_tab1_predict_button"):
                # Create input DataFrame with correct feature order
                row = {col: feature_values.get(col, np.nan) for col in all_features}
                X_input = pd.DataFrame([row], columns=all_features)

                # Convert categorical features to category dtype
                for col in categorical_features:
                    X_input[col] = X_input[col].astype("string").astype("category")

                # Convert numeric features to numeric dtype
                non_cat_cols = [c for c in all_features if c not in categorical_features]
                X_input[non_cat_cols] = X_input[non_cat_cols].apply(pd.to_numeric, errors="coerce")

                # Make prediction (ensure non-negative)
                pred = float(booster.predict(X_input)[0])
                pred = max(pred, 0.0)
                st.success(f"Predicted demand: **{pred:.2f} trips**")

                # Show feature details in expander
                with st.expander("Show engineered features sent to the model"):
                    st.json(feature_values)

    # ===============================================
    # TAB 2: Prediction for a Day
    # ===============================================
    with tab2:
        st.subheader("Predict demand for all hours of a day")
        
        # Station selection dropdown
        station_input_tab2 = st.selectbox(
            "Station", station_list, key="page1_tab2_station"
        )

        # Date and Holiday on the same row
        col_date2, col_holiday2 = st.columns(2)
        with col_date2:
            date_input_tab2 = st.date_input(
                "Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="page1_tab2_date",
                help=f"Weather data available from {min_date} to {max_date}"
            )
        with col_holiday2:
            is_holiday_input_tab2 = st.selectbox(
                "Holiday (0=No, 1=Yes)", options=[0, 1], index=0, key="page1_tab2_hol"
            )

        # Predict button
        if st.button("Predict demand", key="page1_tab2_predict_button"):
            # Get weather data for the entire day
            day_weather_df = get_weather_for_day(weather_df, date_input_tab2)
            
            if day_weather_df is None or day_weather_df.empty:
                st.warning(f"No weather data available for {date_input_tab2}.")
            else:
                # Prepare to collect predictions for each hour
                hourly_predictions = []
                hourly_temperatures = []
                hours_list = []

                all_features = meta["all_features"]
                categorical_features = meta["categorical_features"]

                # Loop through all 24 hours
                for hour in range(24):
                    # Get weather for this hour
                    hour_row = day_weather_df[day_weather_df["hour"] == hour]
                    
                    if hour_row.empty:
                        # Skip hours without weather data
                        continue
                    
                    hour_row = hour_row.iloc[0]
                    
                    # Build features for this hour
                    feature_values = build_features(
                        station=station_input_tab2,
                        date=date_input_tab2,
                        hour=hour,
                        is_holiday=int(is_holiday_input_tab2),
                        temperature=hour_row["temperature"],
                        feels_like=hour_row["feels_like"],
                        wind_speed=hour_row["wind_speed"],
                        bad_weather=hour_row["bad_weather"],
                        meta=meta,
                    )

                    # Create input DataFrame
                    row = {col: feature_values.get(col, np.nan) for col in all_features}
                    X_input = pd.DataFrame([row], columns=all_features)

                    # Convert categorical features to category dtype
                    for col in categorical_features:
                        X_input[col] = X_input[col].astype("string").astype("category")

                    # Convert numeric features to numeric dtype
                    non_cat_cols = [c for c in all_features if c not in categorical_features]
                    X_input[non_cat_cols] = X_input[non_cat_cols].apply(pd.to_numeric, errors="coerce")

                    # Make prediction (ensure non-negative)
                    pred = float(booster.predict(X_input)[0])
                    pred = max(pred, 0.0)

                    hours_list.append(hour)
                    hourly_predictions.append(pred)
                    hourly_temperatures.append(hour_row["temperature"])

                # Create DataFrame for plotting
                chart_df = pd.DataFrame({
                    "Hour": hours_list,
                    "Predicted Demand": [round(p, 2) for p in hourly_predictions],
                    "Temperature (°C)": hourly_temperatures
                })

                # Display the dual-axis line chart
                st.markdown(f"### Hourly Predictions for {date_input_tab2}")
                
                # Create a dual-axis chart using Plotly
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Add Predicted Demand line
                fig.add_trace(
                    go.Scatter(
                        x=chart_df["Hour"],
                        y=chart_df["Predicted Demand"],
                        name="Demand",
                        line=dict(color="red", width=2),
                        mode="lines"
                    ),
                    secondary_y=False,
                )

                # Add Temperature line
                fig.add_trace(
                    go.Scatter(
                        x=chart_df["Hour"],
                        y=chart_df["Temperature (°C)"],
                        name="Temperature",
                        line=dict(color="pink", width=2),
                        mode="lines"
                    ),
                    secondary_y=True,
                )

                # Update layout and axis labels
                fig.update_layout(
                    title_text="Hourly Demand Prediction and Temperature",
                    xaxis_title="Hour of Day",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified"
                )

                # Set y-axis
                fig.update_yaxes(title_text="Predicted Demand", showgrid=False, secondary_y=False)
                fig.update_yaxes(title_text="Temperature (°C)", showgrid=False, secondary_y=True)

                # Set x-axis
                fig.update_xaxes(
                    tickmode="linear",
                    tick0=0,
                    dtick=1,
                    range=[-0.5, 23.5]
                )

                # Display the Plotly chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)

                # Also show raw data in an expander
                with st.expander("Show hourly data table"):
                    st.dataframe(chart_df, use_container_width=True)


# =================================================
# PAGE 2: Demand Forecast with Custom Inputs
# =================================================
def render_page2():
    # Load model and metadata
    booster, meta = load_model_assets()

    st.header("Station Demand Forecast with Custom Inputs")
    st.write(
        "Manually enter weather conditions and other parameters to predict "
        "bike demand for any future date."
    )

    # Get station list from metadata
    station_list = meta["station"]
    
    # Station selection dropdown
    station_input = st.selectbox("Station", station_list, key="page2_station")

    # Date and Holiday
    min_date = dt.date(2025, 12, 26)
    max_date = dt.date(2040, 12, 31)
    
    col_date, col_holiday = st.columns(2)
    with col_date:
        date_input = st.date_input(
            "Date",
            value=dt.date(2026, 1, 1),
            min_value=min_date,
            max_value=max_date,
            key="page2_date",
        )
    with col_holiday:
        is_holiday_input = st.selectbox(
            "Holiday (0=No, 1=Yes)", options=[0, 1], index=0, key="page2_hol"
        )

    # Hour slider
    hour_input = st.slider("Hour of day", 0, 23, 12, key="page2_hour")

    # Weather inputs
    st.markdown("### Weather Conditions")
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        temperature_input = st.number_input(
            "Temperature (°C)", -40.0, 50.0, 20.0, key="page2_temp"
        )
    with w_col2:
        feels_like_input = st.number_input(
            "Feels like (°C)", -40.0, 50.0, 20.0, key="page2_feels",
        )
    with w_col3:
        wind_speed_input = st.number_input(
            "Wind speed (km/h)", 0.0, 80.0, 10.0, key="page2_wind"
        )
    with w_col4:
        bad_weather_input = st.selectbox(
            "Bad weather (0=No, 1=Yes)", options=[0, 1], index=0, key="page2_bad_weather"
        )

    # Build features for prediction
    feature_values = build_features(
        station=station_input,
        date=date_input,
        hour=hour_input,
        is_holiday=int(is_holiday_input),
        temperature=temperature_input,
        feels_like=feels_like_input,
        wind_speed=wind_speed_input,
        bad_weather=int(bad_weather_input),
        meta=meta,
    )

    all_features = meta["all_features"]
    categorical_features = meta["categorical_features"]

    # Predict button
    if st.button("Predict demand", key="page2_predict_button"):
        # Create input DataFrame with correct feature order
        row = {col: feature_values.get(col, np.nan) for col in all_features}
        X_input = pd.DataFrame([row], columns=all_features)

        # Convert categorical features to category dtype
        for col in categorical_features:
            X_input[col] = X_input[col].astype("string").astype("category")

        # Convert numeric features to numeric dtype
        non_cat_cols = [c for c in all_features if c not in categorical_features]
        X_input[non_cat_cols] = X_input[non_cat_cols].apply(pd.to_numeric, errors="coerce")

        # Make prediction (ensure non-negative)
        pred = float(booster.predict(X_input)[0])
        pred = max(pred, 0.0)
        st.success(f"Predicted demand: **{pred:.2f} trips**")

        # Show feature details in expander
        with st.expander("Show engineered features sent to the model"):
            st.json(feature_values)


# =================================================
# PAGE 3: Station Demand Clusters
# =================================================
@st.cache_data
def load_station_clusters():
    return pd.read_csv("station_clusters.csv")


def render_page3():
    # Load cluster data
    clusters_df = load_station_clusters()

    # Validate required columns exist
    required_cols = {"station", "lat", "lon", "mean_demand", "cluster_label"}
    missing = required_cols - set(clusters_df.columns)
    if missing:
        st.error(f"Missing columns in station_clusters.csv: {missing}")
        return

    st.header("Station Demand Clusters")

    st.write(
        "Stations are grouped into **3 clusters** based on their average hourly demand in 2024: "
        "`low`, `medium`, and `high`. The heatmap below shows where stations in "
        "each cluster are located across the city."
    )

    # -----------------------------------------------
    # Cluster filter selection
    # -----------------------------------------------
    cluster_choice = st.radio(
        "Which cluster do you want to visualize?",
        ["all", "low", "medium", "high"],
        index=0,
        key="cluster_choice",
    )

    # Filter data based on selection
    if cluster_choice == "all":
        df_plot = clusters_df.copy()
        st.subheader("All clusters")
    else:
        df_plot = clusters_df[clusters_df["cluster_label"] == cluster_choice].copy()
        st.subheader(f"Cluster: {cluster_choice} demand")

    st.caption(f"Stations shown: {len(df_plot)}")

    # Assign colors to each cluster for visualization
    color_map = {
        "low": [0, 120, 255],      # blue-ish
        "medium": [255, 215, 0],   # yellow-ish
        "high": [255, 0, 0],       # red
    }
    df_plot["color"] = df_plot["cluster_label"].map(color_map)

    # Create PyDeck map with heatmap and scatter layers
    if len(df_plot) > 0:
        # Set initial view centered on the data
        view_state = pdk.ViewState(
            latitude=df_plot["lat"].mean(),
            longitude=df_plot["lon"].mean(),
            zoom=11,
            pitch=45,
        )

        # Heatmap layer showing demand intensity
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=df_plot,
            get_position="[lon, lat]",
            get_weight="mean_demand",
            radiusPixels=60,
            aggregation="SUM",
        )

        # Scatter layer showing individual station points
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_plot,
            get_position="[lon, lat]",
            get_radius=40,
            get_fill_color="color",
            pickable=True,
        )

        # Tooltip for station info on hover
        tooltip = {
            "html": (
                "<b>Station:</b> {station}<br/>"
                "<b>Cluster:</b> {cluster_label}<br/>"
                "<b>Mean demand:</b> {mean_demand}"
            ),
            "style": {"color": "white"},
        }

        # Combine layers into deck
        deck = pdk.Deck(
            layers=[heatmap_layer, scatter_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
        )

        st.pydeck_chart(deck)
    else:
        st.warning("No stations to display for this cluster filter.")

    # Display cluster summary statistics
    st.markdown("### Cluster summary")
    summary = (
        clusters_df.groupby("cluster_label")["mean_demand"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "cluster_label": "Cluster Label",
                "count": "Station Count",
                "mean": "Avg Demand",
                "min": "Min Demand",
                "max": "Max Demand",
            }
        )
        .sort_values("Avg Demand")
    )
    summary[summary.select_dtypes("number").columns] = summary.select_dtypes("number").round(2)
    st.dataframe(summary, use_container_width=True)


# =================================================
# SIDEBAR ROUTER
# =================================================
st.sidebar.title("Select view:")
view = st.sidebar.radio(
    "",
    [
        "16-Day Demand Forecast",
        "Demand Forecast with Custom Inputs",
        "Station Demand Clusters",
    ],
    index=0,
)

if view == "16-Day Demand Forecast":
    render_page1()
elif view == "Demand Forecast with Custom Inputs":
    render_page2()
else:
    render_page3()

# Global footer
st.divider()
st.caption(
    "Demand is defined as the sum of departures and returns. "
    "Use 2024 data as the baseline for prediction and clustering, "
    "and analyze only the top 400 stations by demand."
)
