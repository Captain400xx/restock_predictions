# ----------------------------------------------------------------------
# Pokémon Card Drop Forecast - v7.6 (15-min precision + time-weighting + local tz)
# ----------------------------------------------------------------------

# -------------------------
# 1. Import Libraries
# -------------------------
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import cmdstanpy
print("✅ Prophet backend:", cmdstanpy.__version__)
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pytz
from io import StringIO
import warnings
from datetime import datetime, timedelta
import numpy as np
import os
import streamlit.components.v1 as components


# Ignore warnings
warnings.filterwarnings("ignore")

# -------------------------
# 2. App Configuration & Initial Setup
# -------------------------
st.set_page_config(
    page_title="RestockR Predictions",
    page_icon="logo2.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

EASTERN = pytz.timezone('US/Eastern')
DATA_FILE = "my_restock_data.csv"

# -------------------------
# 3. Load Your Data from CSV
# -------------------------
@st.cache_data
def load_main_data(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if 'DateTime' not in df.columns or 'Retailer' not in df.columns:
                st.error(f"Error: Your CSV file '{file_path}' must contain 'DateTime' and 'Retailer' columns.")
                return None
            # return CSV string (same as before)
            return df.to_csv(index=False)
        except Exception as e:
            st.error(f"Error reading your CSV file: {e}")
            return None
    else:
        st.warning(f"File '{file_path}' not found. Using sample data.")
        default_csv_data = """DateTime,Retailer
2025-09-01 10:05,Pokemon Center
2025-09-02 14:10,Walmart
2025-09-03 11:30,Target
"""
        return default_csv_data

# -------------------------
# 4. Feature Engineering & Helper Functions
# -------------------------
@st.cache_data
def create_features_for_ml(df):
    # expects df with DateTime and Count already present; DateTime may be index or column
    if 'DateTime' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'DateTime'})
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)

    # basic time fields (15-min precision)
    df['hour'] = df['DateTime'].dt.hour
    df['minute'] = df['DateTime'].dt.minute
    df['quarter'] = (df['minute'] // 15).astype(int)  # 0..3
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['dayofyear'] = df['DateTime'].dt.dayofyear
    df['weekofyear'] = df['DateTime'].dt.isocalendar().week.astype(int)
    df['month'] = df['DateTime'].dt.month

    # feature: minutes since last restock event
    df['minutes_since_last'] = df['DateTime'].diff().dt.total_seconds().div(60).fillna(0)



    # continuous time-of-day feature in minutes for smooth cyclical encodings
    df['minutes_of_day'] = df['hour'] * 60 + df['minute']
    df['tod_sin'] = np.sin(2 * np.pi * df['minutes_of_day'] / (24 * 60.0))
    df['tod_cos'] = np.cos(2 * np.pi * df['minutes_of_day'] / (24 * 60.0))

    # day-of-week cyclical
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

    # lag features for 15-min resolution:
    # shift(1) => previous 15 min, shift(4) => previous 1 hour, shift(96) => previous 24 hours
    df['lag_15m'] = df['Count'].shift(1).fillna(0)
    df['lag_1h'] = df['Count'].shift(4).fillna(0)
    df['lag_1d'] = df['Count'].shift(96).fillna(0)
    df['lag_1w'] = df['Count'].shift(96 * 7).fillna(0)

    # rolling mean features to capture short-term trends
    df['rolling_1h_mean'] = df['Count'].rolling(4, min_periods=1).mean()
    df['rolling_3h_mean'] = df['Count'].rolling(12, min_periods=1).mean()
    df['rolling_1d_mean'] = df['Count'].rolling(96, min_periods=1).mean()


    # fill any remaining NaNs
    df.fillna(0, inplace=True)
    return df

@st.cache_data
def process_data_for_forecasting(csv_data_string):
    df = pd.read_csv(StringIO(csv_data_string))
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    # Each row = 1 recorded event
    df['Count'] = 1

    # Resample to 15-minute intervals per retailer
    df_15min = (
        df.set_index('DateTime')
          .groupby('Retailer')
          .resample('15T')
          .sum(numeric_only=True)
          .reset_index()
    )

    all_retailers = df_15min["Retailer"].unique()
    processed_dfs = []
    for r in all_retailers:
        retailer_df = df_15min[df_15min["Retailer"] == r].copy()
        # ensure continuous 15-min index over the observed range
        full_range = pd.date_range(start=retailer_df['DateTime'].min(), end=retailer_df['DateTime'].max(), freq='15T')
        retailer_df = retailer_df.set_index('DateTime').reindex(full_range).fillna(0).reset_index().rename(columns={'index': 'DateTime'})
        retailer_df['Retailer'] = r
        retailer_featured_df = create_features_for_ml(retailer_df)
        processed_dfs.append(retailer_featured_df)
    if processed_dfs:
        return pd.concat(processed_dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=['DateTime', 'Retailer', 'Count'])

def filter_by_time(df, future_only=True):
    # Ensure ds is timezone-aware in UTC for robust comparisons
    if df is None or df.empty:
        return df
    df = df.copy()
    if 'ds' in df.columns:
        # if naive, assume UTC
        if df['ds'].dt.tz is None:
            df['ds'] = df['ds'].dt.tz_localize(pytz.UTC)
    now_utc = datetime.now(pytz.UTC)
    if future_only:
        return df[df['ds'] >= now_utc]
    return df

# --- Detect and apply local timezone automatically ---
def convert_to_local_time(df, time_col='ds', tz_str='UTC'):
    """
    Convert df[time_col] (which should be UTC-aware or naive UTC) to the given tz_str (pytz name).
    Returns a copy with time_col converted to tz-aware times in tz_str.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    try:
        # ensure it's datetime
        df[time_col] = pd.to_datetime(df[time_col])
        # if naive, assume UTC
        if df[time_col].dt.tz is None:
            df[time_col] = df[time_col].dt.tz_localize(pytz.UTC)
        # convert to requested timezone
        local_tz = pytz.timezone(tz_str)
        df[time_col] = df[time_col].dt.tz_convert(local_tz)
    except Exception:
        # best-effort fallback: keep as naive datetimes
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception:
            pass
    return df


# ----- MODEL TRAINING FUNCTIONS -----
@st.cache_data
def train_prophet_model(data, forecast_horizon):
    # data: DataFrame for single retailer with DateTime and Count at 15-min intervals
    df_prophet = data.rename(columns={"DateTime": "ds", "Count": "y"})[['ds', 'y']].copy()
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    # Prophet expects naive times; we'll treat them as UTC for consistent handling
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(pytz.UTC).dt.tz_convert(pytz.UTC).dt.tz_localize(None)

    # --- IMPROVED Prophet Model SETUP ---
    # Step 1: Smooth data slightly for better signal
    df_prophet['y'] = df_prophet['y'].rolling(2, min_periods=1).mean()

    # Step 2: Define known special events (optional – you can add real Pokémon launch dates here)
    holidays = pd.DataFrame({
        'holiday': 'release_day',
        'ds': pd.to_datetime([
            '2025-02-09',  # example release dates
            '2025-05-10',
            '2025-08-20'
        ])
    })

    # Step 3: Initialize Prophet with better tuning
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=holidays,
        changepoint_prior_scale=0.15,     # allows moderate flexibility
        seasonality_prior_scale=10.0,     # stronger cycles
        interval_width=0.9
    )

    # Step 4: Add finer seasonality cycles
    model.add_seasonality(name='hourly', period=1, fourier_order=10)
    model.add_seasonality(name='3hourly', period=3, fourier_order=8)



    model.fit(df_prophet)
    # forecast_horizon is in days; convert to number of 15-min periods
    periods = forecast_horizon * 24 * 4
    future = model.make_future_dataframe(periods=periods, freq="15T")
    forecast = model.predict(future)
    # ensure ds is timezone-aware UTC (for consistent downstream conversion)
    forecast['ds'] = pd.to_datetime(forecast['ds']).dt.tz_localize(pytz.UTC)
    forecast['Weekday'] = forecast['ds'].dt.tz_convert(pytz.UTC).dt.day_name()
    # Prophet returns 'yhat' as predictions; normalize to positive
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    return model, forecast

@st.cache_data
def train_ml_model(data, forecast_horizon, model_type='lgbm'):
    # data: features created by create_features_for_ml
    FEATURES = ['hour', 'quarter', 'dayofweek', 'dayofyear', 'weekofyear', 'month',
                'tod_sin', 'tod_cos', 'dayofweek_sin', 'dayofweek_cos',
                'is_weekend', 'is_business_hours', 'lag_15m', 'lag_1h', 'lag_1d', 'lag_1w']
    TARGET = 'Count'

    data = data.copy()
    X_train = data[FEATURES]
    y_train = data[TARGET]

    # --- Time-weighted training ---
    # More recent observations should have higher weight.
    # compute hours difference from most recent time
    if 'DateTime' in data.columns:
        max_time = data['DateTime'].max()
        data['time_diff_hours'] = (max_time - data['DateTime']).dt.total_seconds() / 3600.0
    else:
        data['time_diff_hours'] = 0.0
    # exponential decay weights; decay_factor controls how fast older data down-weighted
    decay_factor = 0.002  # tweak as needed: smaller -> slower decay
    sample_weights = np.exp(-decay_factor * data['time_diff_hours'])

    # --- Model selection ---
    if model_type == 'lgbm':
        model = lgb.LGBMRegressor(random_state=42, objective='poisson')
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(random_state=42, objective='count:poisson')
    else:
        model = cb.CatBoostRegressor(random_state=42, loss_function='Poisson', verbose=0)

    # Fit with sample weights
    try:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    except TypeError:
        # fallback for models that expect different param name
        model.fit(X_train, y_train)

    # Build future dates at 15-min resolution
    last_date = data['DateTime'].max()
    # forecast_horizon is in days: multiply by 96 to get 15-min periods
    periods = forecast_horizon * 24 * 4
    future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=15), periods=periods, freq='15T')
    future_df = pd.DataFrame({'DateTime': future_dates})
    # create features for both history + future so lag features exist
    full_history = pd.concat([data.set_index('DateTime'), future_df.set_index('DateTime')], axis=0, sort=False).reset_index()
    full_history.rename(columns={'index': 'DateTime'}, inplace=True)
    full_history['Count'] = full_history['Count'].fillna(0)
    full_history_featured = create_features_for_ml(full_history)
    X_future = full_history_featured.iloc[-len(future_df):][FEATURES]
    yhat = model.predict(X_future)
    future_df['yhat'] = np.clip(yhat, 0, None)
    forecast_df = future_df.rename(columns={'DateTime': 'ds'})
    # set ds timezone-aware in UTC
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.tz_localize(pytz.UTC)
    forecast_df['Weekday'] = forecast_df['ds'].dt.tz_convert(pytz.UTC).dt.day_name()
    return model, forecast_df

# ----- ANALYSIS & PLOTTING -----
def get_big_restocks(forecast_df):
    if forecast_df is None or forecast_df.empty:
        return 0.0, pd.DataFrame()
    forecast_df = forecast_df.copy()
    forecast_df['yhat'] = forecast_df.get('yhat', forecast_df.get('yhat', 0)).clip(lower=0)
    cutoff = forecast_df["yhat"].quantile(0.90) if "yhat" in forecast_df.columns else 0.0
    cutoff = max(cutoff, 0.5)
    big = forecast_df[forecast_df["yhat"] >= cutoff].copy()
    return cutoff, big.sort_values("ds")

def create_forecast_chart(forecast_df, big_restocks_df, retailer, title, color):
    fig = go.Figure()
    if forecast_df is None or forecast_df.empty:
        fig.update_layout(title=f"No forecast available for {retailer}")
        return fig

    # Ensure ds is tz-aware for display
    f = forecast_df.copy()
    if f['ds'].dt.tz is None:
        f['ds'] = f['ds'].dt.tz_localize(pytz.UTC)
    
    display_times = f['ds'].dt.tz_convert(pytz.timezone(selected_tz))


    fig.add_trace(go.Scatter(x=display_times, y=f["yhat"], mode="lines", name="Forecast",
                             line=dict(width=2, color=color),
                             hovertemplate='%{x|%Y-%m-%d %H:%M} — %{y:.1f} activity<extra></extra>'))
    if big_restocks_df is not None and not big_restocks_df.empty:
        big = big_restocks_df.copy()
        if big['ds'].dt.tz is None:
            big['ds'] = big['ds'].dt.tz_localize(pytz.UTC)
        big_display = big['ds'].dt.tz_convert(pytz.timezone(selected_tz))
        fig.add_trace(go.Scatter(x=big_display, y=big['yhat'], mode="markers", name="Predicted Big Restock",
                                 marker=dict(size=10, color='red', symbol='star'),
                                 hovertemplate='%{x|%Y-%m-%d %H:%M} — %{y:.1f} activity<extra></extra>'))

    now_local = datetime.now(pytz.UTC).astimezone(pytz.timezone(selected_tz))
    ymax = max(f['yhat'].max() * 1.2, 5) if not f.empty else 5
    fig.update_layout(
        title=f"{title} for {retailer}",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        yaxis=dict(title="Predicted Restock Activity", range=[0, ymax]),
        xaxis=dict(title=f"Date (Local: {selected_tz})", showgrid=True),
        shapes=[dict(type='line', x0=now_local, y0=0, x1=now_local, y1=1, yref='paper', line=dict(color='RoyalBlue', width=2, dash='dash'))],
        annotations=[dict(x=now_local, y=1.05, yref='paper', text='Current Time', showarrow=False)]
    )
    return fig

def display_consensus_schedule(df):
    st.markdown("""<style>.schedule-table { width: 100%; border-collapse: collapse; font-size: 0.9em; } .schedule-table th, .schedule-table td { padding: 8px; text-align: left; border-bottom: 1px solid #444; } .schedule-table th { background-color: #1a1a1a; } .high-confidence { background-color: rgba(40, 167, 69, 0.3); } .medium-confidence { background-color: rgba(255, 193, 7, 0.3); } .low-confidence { background-color: rgba(220, 53, 69, 0.3); } .day-group { font-weight: bold; font-size: 1.1em; padding-top: 15px; }</style>""", unsafe_allow_html=True)
    html = "<table class='schedule-table'>"
    html += "<tr><th>Time (Local)</th><th>Avg. Activity</th><th>Confidence</th><th>Models in Agreement</th></tr>"
    if df.empty:
        html += "<tr><td colspan='4' style='text-align:center;'>No significant restocks predicted.</td></tr>"
    else:
        df_sorted = df.sort_values('time_group')
        current_day = None
        for _, row in df_sorted.iterrows():
            row_date = row['time_group'].date()
            if row_date != current_day:
                current_day = row_date
                day_str = row['time_group'].strftime('%A, %b %d')
                html += f"<tr><td colspan='4' class='day-group'>{day_str}</td></tr>"
            confidence_class = f"{row['Confidence'].lower()}-confidence"
            time_str = row['time_group'].strftime('%I:%M %p')
            avg_activity_str = f"{row['avg_activity']:.1f}"
            html += f"<tr class='{confidence_class}'><td>{time_str}</td><td>{avg_activity_str}</td><td>{row['Confidence']}</td><td>{row['models']}</td></tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

def create_consensus_chart(df):
    if df.empty:
        return go.Figure().update_layout(title="No significant activity predicted.")

    color_map = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    df = df.copy()
    df['color'] = df['Confidence'].map(color_map)

    # --- Create hierarchical labels ---
    # Level 1 (Top label): "Tue, Oct 21"
    df['day_label_L1'] = df['time_group'].dt.strftime('%a, %b %d')
    # Level 2 (Bottom label): "02:00 PM"
    df['day_label_L2'] = df['time_group'].dt.strftime('%I:%M %p')
    # --- End label creation ---

    fig = go.Figure(go.Bar(
        # Pass BOTH columns to 'x' to create the hierarchy
        x=[df['day_label_L1'], df['day_label_L2']], 
        y=df['avg_activity'], 
        marker_color=df['color'],
        # Update hovertemplate to show both parts of the x-axis
        hovertemplate="<b>%{x[0]}<br>%{x[1]}</b><br>Confidence: %{customdata[0]}<br>Avg. Activity: %{y:.1f}<extra></extra>",
        customdata=df[['Confidence']]
    ))

    fig.update_layout(
        title="Visual Prediction Schedule", 
        xaxis_title="Predicted Event Time (Local)", 
        yaxis_title="Average Predicted Activity", 
        template="plotly_white", 
        height=400,
        # This helps keep the labels clean
        xaxis=dict(tickfont=dict(size=10)) 
    )
    return fig

def create_importance_chart(df):
    fig = go.Figure(go.Bar(x=df['Importance'], y=df['Feature'], orientation='h'))
    fig.update_layout(title="Model Feature Importance (All Models)", xaxis_title="Average Importance", yaxis_title="Feature", template="plotly_white", yaxis=dict(autorange="reversed"), height=400)
    return fig

def create_calendar_heatmap(forecast_df, retailer):
    if forecast_df is None or forecast_df.empty:
        return go.Figure().update_layout(title="No data")
    df = forecast_df.copy()
    # convert to local for nicer date labels
    try:
        display_times = df['ds'].dt.tz_convert(pytz.timezone(selected_tz))
    except Exception:
        pass

    if df['ds'].dt.tz is None:
        df['ds'] = df['ds'].dt.tz_localize(pytz.UTC)
    df['local_ds'] = df['ds'].dt.tz_convert(pytz.timezone(selected_tz))
    df['date'] = df['local_ds'].dt.date
    daily_max = df.groupby('date')['yhat'].max().reset_index()
    daily_max['day_of_week'] = pd.to_datetime(daily_max['date']).dt.dayofweek
    daily_max['week_of_year'] = pd.to_datetime(daily_max['date']).dt.isocalendar().week
    daily_max['day_str'] = pd.to_datetime(daily_max['date']).dt.strftime('%a<br>%d')
    weeks = sorted(daily_max['week_of_year'].unique())
    heatmap_data = np.full((7, len(weeks)), np.nan)
    text_data = np.full((7, len(weeks)), '', dtype=object)
    for _, row in daily_max.iterrows():
        try:
            week_idx = weeks.index(row['week_of_year'])
            day_idx = row['day_of_week']
            heatmap_data[day_idx, week_idx] = row['yhat']
            text_data[day_idx, week_idx] = row['day_str']
        except (IndexError, ValueError):
            continue
    fig = go.Figure(data=go.Heatmap(z=heatmap_data, x=[f"Week {w}" for w in weeks], y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], hoverongaps=False, text=text_data, texttemplate="%{text}", colorscale='Oranges', colorbar_title='Max Activity'))
    fig.update_layout(title=f"Daily Activity Heatmap for {retailer}", xaxis_title="Week of the Year", height=400)
    return fig

def create_hourly_heatmap(forecast_df, retailer):
    if forecast_df is None or forecast_df.empty:
        return go.Figure().update_layout(title="No data")
    df = forecast_df.copy()
    # convert to local for consistent hour labels
    try:
        display_times = df['ds'].dt.tz_convert(pytz.timezone(selected_tz))
    except Exception:
        pass

    if df['ds'].dt.tz is None:
        df['ds'] = df['ds'].dt.tz_localize(pytz.UTC)
    df['local_ds'] = df['ds'].dt.tz_convert(pytz.timezone(selected_tz))
    df['hour'] = df['local_ds'].dt.hour
    df['dayofweek'] = df['local_ds'].dt.dayofweek
    hourly_avg = df.groupby(['dayofweek', 'hour'])['yhat'].mean().reset_index()
    heatmap_data = hourly_avg.pivot(index='dayofweek', columns='hour', values='yhat').reindex(index=range(7), columns=range(24))
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hour_labels = [f"{h}:00" for h in range(24)]
    fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=hour_labels, y=day_labels, colorscale='Oranges', colorbar_title='Avg. Activity'))
    fig.update_layout(title=f"Hourly Activity Heatmap for {retailer}", xaxis_title="Hour of Day (Local)", yaxis_title="Day of Week", height=500)
    return fig

# -------------------------
# 5. Main App Logic
# -------------------------
col1, col2 = st.columns([1, 8])
with col1:
    if os.path.exists("logo2.png"):
        st.image("logo2.png", width=100)
with col2:
    st.markdown(
        """
        <h1 style='font-size:48px; font-weight:900;'>
            <span style='color:#00C46A;'>Restock</span><span style='color:#FF4B4B;'>R</span>
            <span style='color:#00C46A;'> Predictions</span>
        </h1>
        """,
        unsafe_allow_html=True
    )




raw_data_string = load_main_data(DATA_FILE)
if raw_data_string is None:
    st.stop()
full_df = process_data_for_forecasting(raw_data_string)

with st.expander("👋 How to Use This App", expanded=True):
    st.markdown("""
    This app forecasts Pokémon card restock activity using multiple models.
    - **Select a Retailer:** Use the dropdown in the sidebar.
    - **Set the Forecast Horizon:** Use the slider to decide how far into the future to predict. Up to 30 days
     - **Explore the Tabs:**
        - **⭐ Prediction Schedule:** A visual schedule of high-confidence predictions.
        - **📅 2-Week View:** A chart to check model accuracy against the past week's data.
        - **📊 Analysis:** Heatmaps showing the hottest days and times for restocks.
        - **Model Tabs (Prophet, etc.):** Dive deep into the forecast of each individual model.


         **What is 'Predicted Restock Activity'?** Since we are counting every restock event as '1', the model forecasts the **frequency of restock events per hour**.
         - A **low** score (e.g., 0-1) means no or very few restock events are expected.
         - A **high** score (e.g., 5+) means the model predicts a cluster of many separate restock events in that hour.

    Note: Predictions are now at 15-minute precision. Models are time-weighted to emphasize recent data.
    """)

# -------------------------
# 6. Sidebar Controls
# -------------------------
st.markdown("""<style>[data-testid="stSidebar"] > div:first-child {padding-top: 1rem;}</style>""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Controls")
retailers = sorted(full_df["Retailer"].unique().tolist())
if not retailers:
    st.error("No retailers found.")
    st.stop()

default_index = 0
if 'Target' in retailers:
    default_index = retailers.index('Target')
retailer = st.sidebar.selectbox("Choose a retailer to forecast", retailers, index=default_index, label_visibility="collapsed")

forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 14, 1)
forecast_hours = forecast_horizon * 24

st.sidebar.markdown("---")
st.sidebar.subheader("Data Summary")
st.sidebar.markdown("**Next High-Confidence Alert:**")
countdown_placeholder = st.sidebar.empty()
retailer_history_df = full_df[full_df['Retailer'] == retailer]
if retailer_history_df.empty:
    st.sidebar.info("Selected retailer has no data.")
min_date = retailer_history_df['DateTime'].min().strftime('%b %d, %Y') if not retailer_history_df.empty else "N/A"
max_date = retailer_history_df['DateTime'].max().strftime('%b %d, %Y') if not retailer_history_df.empty else "N/A"
event_count = int(retailer_history_df['Count'].sum()) if not retailer_history_df.empty else 0
st.sidebar.markdown(f"""<div style="font-size: 0.9em;">Data Available From: <strong style="font-size: 1.1em;">{min_date}</strong><br>Data Available To: <strong style="font-size: 1.1em;">{max_date}</strong><br>Total Recorded Events: <strong style="font-size: 1.1em;">{event_count:,}</strong></div>""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Chart Customization")

chart_color = "#FFA500"  # default orange/yellow

# -------------------------
# Sidebar timezone selector (add this near your other sidebar controls)
# -------------------------
import pytz

COMMON_TZ = [
    "UTC",
    "US/Eastern",
    "US/Central",
    "US/Mountain",
    "US/Pacific",
    "Europe/London",
    "Europe/Berlin",
    "Asia/Tokyo",
    "Australia/Sydney"
]
# default to Eastern
default_tz = "US/Eastern"
# put user's tz selection into a variable used app-wide
selected_tz = st.sidebar.selectbox("Display timezone for charts & labels", COMMON_TZ, index=COMMON_TZ.index(default_tz))
st.sidebar.caption(f"Selected timezone: {selected_tz}")


st.sidebar.markdown("---")
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=100)
st.sidebar.warning("This data is the property of RestockR Monitors. Unauthorized sharing, distribution, or external use of this information may result in penalties or legal action.")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Captain400x**")

retailer_df = full_df[full_df["Retailer"] == retailer].copy()

# --- Run All Models & Generate Consensus ---
with st.spinner("Training models with advanced features... this may take a moment."):
    prophet_model, prophet_forecast_raw = train_prophet_model(retailer_df, forecast_horizon)
    xgb_model, xgb_forecast_raw = train_ml_model(retailer_df, forecast_horizon, model_type='xgb')
    lgbm_model, lgbm_forecast_raw = train_ml_model(retailer_df, forecast_horizon, model_type='lgbm')
    cat_model, cat_forecast_raw = train_ml_model(retailer_df, forecast_horizon, model_type='catboost')

    FEATURES = ['hour', 'quarter', 'dayofweek', 'dayofyear', 'weekofyear', 'month',
                'tod_sin', 'tod_cos', 'dayofweek_sin', 'dayofweek_cos',
                'is_weekend', 'is_business_hours', 'lag_15m', 'lag_1h', 'lag_1d', 'lag_1w']
    try:
        lgbm_imp = lgbm_model.feature_importances_
        xgb_imp = xgb_model.feature_importances_
        cat_imp = cat_model.feature_importances_
        avg_imp = (lgbm_imp / lgbm_imp.sum() + xgb_imp / xgb_imp.sum() + cat_imp / cat_imp.sum()) / 3
        importance_df = pd.DataFrame({'Feature': FEATURES, 'Importance': avg_imp}).sort_values('Importance', ascending=False)
    except Exception:
        importance_df = pd.DataFrame({'Feature': FEATURES, 'Importance': [1 / len(FEATURES)] * len(FEATURES)}).sort_values('Importance', ascending=False)

# Filter only future predictions (in UTC) and convert to local for display
prophet_forecast = filter_by_time(prophet_forecast_raw)
xgb_forecast = filter_by_time(xgb_forecast_raw)
lgbm_forecast = filter_by_time(lgbm_forecast_raw)
cat_forecast = filter_by_time(cat_forecast_raw)

# Convert all forecast times to viewer's local timezone for display and charting
rophet_forecast = convert_to_local_time(prophet_forecast, time_col='ds', tz_str=selected_tz) if prophet_forecast is not None else prophet_forecast
xgb_forecast = convert_to_local_time(xgb_forecast, time_col='ds', tz_str=selected_tz) if xgb_forecast is not None else xgb_forecast
lgbm_forecast = convert_to_local_time(lgbm_forecast, time_col='ds', tz_str=selected_tz) if lgbm_forecast is not None else lgbm_forecast
cat_forecast = convert_to_local_time(cat_forecast, time_col='ds', tz_str=selected_tz) if cat_forecast is not None else cat_forecast

# But keep copies of UTC forecasts for consensus logic (use UTC internally)
prophet_utc = filter_by_time(prophet_forecast_raw)
xgb_utc = filter_by_time(xgb_forecast_raw)
lgbm_utc = filter_by_time(lgbm_forecast_raw)
cat_utc = filter_by_time(cat_forecast_raw)

_, prophet_big = get_big_restocks(prophet_utc)
_, xgb_big = get_big_restocks(xgb_utc)
_, lgbm_big = get_big_restocks(lgbm_utc)
_, cat_big = get_big_restocks(cat_utc)

all_big_restocks = []
for _, row in prophet_big.iterrows():
    all_big_restocks.append({'ds': row['ds'], 'model': 'Prophet', 'yhat': row['yhat']})
for _, row in xgb_big.iterrows():
    all_big_restocks.append({'ds': row['ds'], 'model': 'XGBoost', 'yhat': row['yhat']})
for _, row in lgbm_big.iterrows():
    all_big_restocks.append({'ds': row['ds'], 'model': 'LightGBM', 'yhat': row['yhat']})
for _, row in cat_big.iterrows():
    all_big_restocks.append({'ds': row['ds'], 'model': 'CatBoost', 'yhat': row['yhat']})

if all_big_restocks:
    consensus_df = pd.DataFrame(all_big_restocks).sort_values('ds').reset_index(drop=True)
    # Round to nearest 3 hours originally; with 15-min precision we group to nearest 15 min *or* keep as-is.
    # We'll round to nearest 15 minutes to group model agreement windows
    consensus_df['time_group'] = consensus_df['ds'].dt.round('15T')
    consensus_summary = consensus_df.groupby('time_group').agg(
        models=('model', lambda x: ', '.join(sorted(x.unique()))),
        model_count=('model', 'nunique'),
        avg_activity=('yhat', 'mean')
    ).reset_index()
    def get_confidence(count):
        if count >= 3:
            return "High"
        if count == 2:
            return "Medium"
        return "Low"
    consensus_summary['Confidence'] = consensus_summary['model_count'].apply(get_confidence)
    # For display, convert the UTC time_group to local tz
    consensus_local = convert_to_local_time(consensus_summary.rename(columns={'time_group': 'ds'}), time_col='ds', tz_str=selected_tz)
    consensus_summary['time_group'] = consensus_local['ds']
    consensus_summary = consensus_summary.sort_values('time_group', ascending=True)
else:
    consensus_summary = pd.DataFrame()

# --- Top Prediction & Countdown ---
st.markdown("---")
st.subheader("🚀 Top Prediction")
if not consensus_summary.empty and "High" in consensus_summary['Confidence'].values:
    top_pred = consensus_summary[consensus_summary['Confidence'] == 'High'].iloc[0]
    time_str = top_pred['time_group'].strftime('%A, %b %d at %I:%M %p')
    st.success(f"**Next High-Confidence Restock Window:** Around **{time_str}**\n- **Models in Agreement:** {top_pred['models']}\n- **Average Predicted Activity:** {top_pred['avg_activity']:.1f}")

    # Countdown: target_time is already localized
    target_time = top_pred['time_group']
    if target_time.tzinfo is None:
        try:
            # assume selected timezone if not already localized
            target_time = pytz.timezone(selected_tz).localize(target_time)
        except Exception:
            # fallback to Eastern if selected_tz is invalid
            target_time = pytz.timezone("US/Eastern").localize(target_time)
    target_timestamp_ms = int(target_time.timestamp() * 1000)

    js_countdown = f"""<h2 id="countdown" style="text-align: left; font-weight: bold; color: #28a745;"></h2><script>var targetTime = {target_timestamp_ms}; function updateCountdown() {{ var now = new Date().getTime(); var diff = targetTime - now; if (diff <= 0) {{ document.getElementById("countdown").innerHTML = "Event in Progress"; clearInterval(interval); return; }} var d = Math.floor(diff / (1000 * 60 * 60 * 24)); var h = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60)); var m = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60)); var s = Math.floor((diff % (1000 * 60)) / 1000); document.getElementById("countdown").innerHTML = d + "d " + h + "h " + m + "m " + s + "s"; }} var interval = setInterval(updateCountdown, 1000); updateCountdown(); </script>"""
    with countdown_placeholder.container():
        components.html(js_countdown, height=75)
else:
    st.info("No 'High Confidence' restocks...")
    countdown_placeholder.info("No high-confidence alert scheduled.")

# -------------------------
# 7. Main Tabs
# -------------------------
tab_names = ["⭐ Prediction Schedule", "📅 2-Week View", "📊 Analysis", "🔮 Prophet", "🚀 LightGBM", "🌟 XGBoost", "🐾 CatBoost"]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.header("Prediction Schedule")
    consensus_fig = create_consensus_chart(consensus_summary)
    st.plotly_chart(consensus_fig, use_container_width=True)
    display_consensus_schedule(consensus_summary)

with tabs[1]:
    st.header("2-Week Accuracy Check (Past vs. Future)")
    st.markdown("This chart shows how well the **Prophet model's forecast (line)** matched the **actual historical data (line)** for the past 7 days.")
    now = datetime.now(pytz.UTC)
    start_date = now - timedelta(days=7)
    end_date = now + timedelta(days=7)
    # convert retailer history (which is in naive DateTime) to UTC-aware for comparison
    history_in_window = retailer_history_df.copy()
    history_in_window['DateTime'] = pd.to_datetime(history_in_window['DateTime'])
    history_in_window = history_in_window[(history_in_window['DateTime'] >= (start_date.astimezone(pytz.UTC).replace(tzinfo=None))) & (history_in_window['DateTime'] <= (now.astimezone(pytz.UTC).replace(tzinfo=None)))]
    prophet_full_forecast = filter_by_time(prophet_forecast_raw, future_only=False)
    # convert prophet_full_forecast to local for plotting
    prophet_full_local = convert_to_local_time(prophet_full_forecast) if prophet_full_forecast is not None else prophet_full_forecast
    forecast_in_window = prophet_full_local[(prophet_full_local['ds'] >= start_date) & (prophet_full_local['ds'] <= end_date)]
    fig_2_week = go.Figure()
    fig_2_week.add_trace(go.Scatter(x=history_in_window['DateTime'], y=history_in_window['Count'], mode='lines', name='Actual Past Activity', line=dict(color='blue', width=2)))
    fig_2_week.add_trace(go.Scatter(x=forecast_in_window['ds'], y=forecast_in_window['yhat'], mode='lines', name='Forecast', line=dict(color=chart_color, width=3, dash='dot')))
    now_local_for_plot = datetime.now(pytz.UTC).astimezone(pytz.timezone(selected_tz))
    fig_2_week.add_shape(type="line", x0=now_local_for_plot, x1=now_local_for_plot, y0=0, y1=1, yref="paper", line=dict(color="red", width=3, dash="dash"))
    fig_2_week.add_annotation(x=now_local_for_plot, y=1.05, yref="paper", text="Current Time", showarrow=False)
    fig_2_week.update_layout(title=f"Past Week vs. Next Week Forecast for {retailer}", xaxis_title="Date", yaxis_title="Restock Activity", template="plotly_white", height=500)
    st.plotly_chart(fig_2_week, use_container_width=True)

with tabs[2]:
    st.header("Analysis Heatmaps")
    st.markdown("These charts show the typical patterns of restock activity, based on model forecasts.")
    st.subheader("Daily Activity Heatmap")
    if prophet_forecast is not None and not prophet_forecast.empty:
        calendar_fig = create_calendar_heatmap(prophet_forecast, retailer)
        st.plotly_chart(calendar_fig, use_container_width=True)
    st.subheader("Hourly Activity Heatmap")
    if prophet_forecast is not None and not prophet_forecast.empty:
        hourly_fig = create_hourly_heatmap(prophet_forecast, retailer)
        st.plotly_chart(hourly_fig, use_container_width=True)
    st.subheader("Model Feature Importance")
    fig_imp = create_importance_chart(importance_df)
    st.plotly_chart(fig_imp, use_container_width=True)

with tabs[3]:
    st.header("🔮 Prophet Model Details")
    fig_prophet = create_forecast_chart(prophet_forecast, prophet_big, retailer, "Prophet Forecast", chart_color)
    st.plotly_chart(fig_prophet, use_container_width=True)
with tabs[4]:
    st.header("🚀 LightGBM Model Details")
    fig_lgbm = create_forecast_chart(lgbm_forecast, lgbm_big, retailer, "LightGBM Forecast", chart_color)
    st.plotly_chart(fig_lgbm, use_container_width=True)
with tabs[5]:
    st.header("🌟 XGBoost Model Details")
    fig_xgb = create_forecast_chart(xgb_forecast, xgb_big, retailer, "XGBoost Forecast", chart_color)
    st.plotly_chart(fig_xgb, use_container_width=True)
with tabs[6]:
    st.header("🐾 CatBoost Model Details")
    fig_cat = create_forecast_chart(cat_forecast, cat_big, retailer, "CatBoost Forecast", chart_color)
    st.plotly_chart(fig_cat, use_container_width=True)

