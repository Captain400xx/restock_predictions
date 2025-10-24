# ----------------------------------------------------------------------
# Pokémon Card Drop Forecast - v7.6 (15-min precision + time-weighting + local tz)
# ----------------------------------------------------------------------

# -------------------------
# 1. Import Libraries
# -------------------------
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

import streamlit as st

# --- Google Analytics Tracking ---
GA_ID = "G-8QM2410BCR"

GA_SCRIPT = f"""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_ID}');
</script>
"""

# Injects the Google Analytics tag into the app (acts like <head>)
st.components.v1.html(GA_SCRIPT, height=0)


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
# ----- MODEL TRAINING FUNCTIONS -----

def get_big_restocks(forecast_df, threshold_factor=1.3):
    """
    Identifies predicted 'big restocks' — points where activity sharply exceeds the trend.
    Returns:
        forecast_df (unchanged)
        big_restocks_df (subset DataFrame of spikes)
    """
    if forecast_df is None or forecast_df.empty:
        return forecast_df, pd.DataFrame()

    df = forecast_df.copy()
    yhat_mean = df['yhat'].mean()
    yhat_std = df['yhat'].std()

    # Define threshold for spikes
    threshold = yhat_mean + threshold_factor * yhat_std
    big_restocks = df[df['yhat'] > threshold]

    return df, big_restocks

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
        changepoint_prior_scale=0.4,     # allows moderate flexibility
        seasonality_prior_scale=15.0,     # stronger cycles
        interval_width=0.95
    )

    # Step 4: Add finer seasonality cycles
    model.add_seasonality(name='hourly', period=1, fourier_order=20)
    model.add_seasonality(name='3hourly', period=3, fourier_order=10)



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
def create_forecast_chart(forecast_df, big_restocks_df, retailer, title, color):
    fig = go.Figure()
    if forecast_df is None or forecast_df.empty:
        fig.update_layout(title=f"No forecast available for {retailer}")
        # Limit x-axis to 7 days from current time
        now_local = datetime.now(pytz.UTC).astimezone(pytz.timezone(selected_tz))
        x_end = now_local + timedelta(days=7)
        fig.update_xaxes(range=[now_local, x_end])

        # Automatically zoom around predicted spikes
        if not forecast_df.empty:
            active = forecast_df[forecast_df['yhat'] > forecast_df['yhat'].mean() * 1.2]
            if not active.empty:
                x_min = active['ds'].min() - pd.Timedelta(hours=12)
                x_max = active['ds'].max() + pd.Timedelta(hours=12)
                fig.update_xaxes(range=[x_min, x_max])
        return fig

    # Ensure ds is tz-aware for display
    f = forecast_df.copy()
    if f['ds'].dt.tz is None:
        f['ds'] = f['ds'].dt.tz_localize(pytz.UTC)
    
    display_times = f['ds'].dt.tz_convert(pytz.timezone(selected_tz))

    # --- Handle extreme spikes so they don't distort the chart ---
    if not f.empty:
        cap_value = f['yhat'].quantile(0.99)  # cap at 99th percentile
        f['yhat_clipped'] = np.clip(f['yhat'], 0, cap_value)
    else:
        f['yhat_clipped'] = f.get('yhat', [])

    # --- Plot forecast line using clipped values ---
    fig.add_trace(go.Scatter(
        x=display_times,
        y=f["yhat_clipped"],
        mode="lines",
        name="Forecast",
        line=dict(width=2, color=color),
        hovertemplate='%{x|%Y-%m-%d %H:%M} — %{y:.1f} activity<extra></extra>'
    ))

    # --- Plot predicted big restocks ---
    if big_restocks_df is not None and not big_restocks_df.empty:
        big = big_restocks_df.copy()
        if big['ds'].dt.tz is None:
            big['ds'] = big['ds'].dt.tz_localize(pytz.UTC)
        big_display = big['ds'].dt.tz_convert(pytz.timezone(selected_tz))
        fig.add_trace(go.Scatter(
            x=big_display,
            y=big['yhat'],
            mode="markers",
            name="Predicted Big Restock",
            marker=dict(size=10, color='red', symbol='star'),
            hovertemplate='%{x|%Y-%m-%d %H:%M} — %{y:.1f} activity<extra></extra>'
        ))

    # --- Layout and time marker ---
    now_local = datetime.now(pytz.UTC).astimezone(pytz.timezone(selected_tz))
    ymax = f['yhat_clipped'].max() * 1.1 if not f.empty else 5
    fig.update_layout(
        title=f"{title} for {retailer}",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        yaxis=dict(title="Predicted Restock Activity", range=[0, ymax]),
        xaxis=dict(title=f"Date (Local: {selected_tz})", showgrid=True),
        shapes=[
            dict(
                type='line',
                x0=now_local, y0=0, x1=now_local, y1=1,
                yref='paper',
                line=dict(color='RoyalBlue', width=2, dash='dash')
            )
        ],
        annotations=[
            dict(x=now_local, y=1.05, yref='paper', text='Current Time', showarrow=False)
        ]
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


def create_hourly_heatmap(forecast_df, retailer):
    import plotly.graph_objects as go
    import pandas as pd

    if forecast_df is None or forecast_df.empty:
        return go.Figure()

    forecast_df['local_time'] = pd.to_datetime(forecast_df['ds'])
    forecast_df['hour'] = forecast_df['local_time'].dt.hour
    forecast_df['day'] = forecast_df['local_time'].dt.day_name().str[:3]

    # Aggregate mean restock activity
    heatmap_data = forecast_df.groupby(['day', 'hour'])['yhat'].mean().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    # Fix color scale (smooth gradient from low -> high)
    color_scale = [
        [0.0, "#e0e1dd"],  # deep navy (low)
        [0.5, "#415a77"],  # mid tone
        [1.0, "#0d1b2a"]   # soft white (high)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=[f"{h%12 or 12}{'a' if h<12 else 'p'}" for h in heatmap_data.columns],
            y=heatmap_data.index,
            colorscale=color_scale,
            zmin=heatmap_data.values.min(),
            zmax=heatmap_data.values.max(),
            colorbar=dict(
                title=dict(
                    text="Restock Intensity",
                    font=dict(size=12, color="white")
                ),
                tickvals=[
                    heatmap_data.values.min(),
                    (heatmap_data.values.min() + heatmap_data.values.max()) / 2,
                    heatmap_data.values.max()
                ],
                ticktext=["Low", "Medium", "High"],
                tickfont=dict(color="white"),
            ),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=f"🕒 Hourly Restock Activity Pattern for {retailer}",
        xaxis=dict(
            tickfont=dict(color="white"),
            title=dict(text="Hour of Day", font=dict(color="white"))
        ),
        yaxis=dict(
            tickfont=dict(color="white"),
            title=dict(text="Day of Week", font=dict(color="white"))
        ),
        title_font=dict(size=18, color="white"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=40, t=80, b=50)
    )

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
    This application analyzes historical Pokémon card restock data and uses multiple machine learning models to forecast future restock activity.

    ### 🧭 Getting Started
    - **Select a Retailer:** Choose from the sidebar dropdown.  
    - **Set Forecast Horizon:** Adjust the slider to define how far ahead to predict (up to 30 days).  
    - **Explore the Tabs:**  
        - ⭐ **Prediction Schedule:** Displays upcoming high-confidence restock windows.  
        - 📊 **Analysis:** Shows heatmaps highlighting the most active days and hours.  
        - 🔮 **Model Tabs:** View detailed forecasts from each individual model.  

    ### 📈 Understanding Predictions
    Each restock event is counted as **1**, so model outputs represent the **expected frequency of restocks per 15-minute interval**.  
    - **Low values (0–1):** Minimal or no restock activity expected.  
    - **High values (5+):** Indicates multiple restocks likely during that time period.  

    ### 🤖 Model Overview
    - **Prophet:** Detects consistent time-based patterns — ideal for recurring events such as nightly or weekly restock cycles. It captures trends like Target’s regular restocks but may smooth out sudden spikes.  
    - **LightGBM:** Reacts quickly to recent restock changes and short-term bursts of activity. Best for spotting sudden shifts in timing, like when a retailer starts restocking earlier than usual.  
    - **XGBoost:** Balances long-term and short-term trends, handling irregular restock patterns across days or weeks. Produces stable, moderate predictions with fewer false alerts.  
    - **CatBoost:** Learns subtle timing differences between retailers and detects patterns hidden within categorical data. Great at recognizing nuanced behaviors, such as retailer-specific restock quirks.  

    *All models are time-weighted to emphasize recent data and improve short-term accuracy.*
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
retailer_history_df = full_df[full_df['Retailer'] == retailer]
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 14, 1)
forecast_hours = forecast_horizon * 24

# -------------------------
# 🧭 Data Summary + Countdown Section
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Summary")

# Basic dataset info
if retailer_history_df.empty:
    st.sidebar.info("Selected retailer has no data.")
else:
    min_date = retailer_history_df['DateTime'].min().strftime('%b %d, %Y')
    max_date = retailer_history_df['DateTime'].max().strftime('%b %d, %Y')
    event_count = int(retailer_history_df['Count'].sum())
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.9em;">
            Data Available From: <strong>{min_date}</strong><br>
            Data Available To: <strong>{max_date}</strong><br>
            Total Recorded Events: <strong>{event_count:,}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
   


# Legend / key
st.sidebar.markdown(
    """
    <div style="font-size: 0.85em; line-height: 1.4;">
        <span style="color:#28a745; font-weight:bold;">🟢 High Confidence</span>: Strong model agreement<br>
        <span style="color:#FFA500; font-weight:bold;">🟠 Medium Confidence</span>: Moderate agreement<br>
        <span style="color:#FF4B4B; font-weight:bold;">🔴 Low Confidence</span>: Weak or single-model signal
    </div>
    """,
    unsafe_allow_html=True,
)

# Add spacing / separator before countdowns
st.sidebar.markdown("<hr style='border:1px solid #333;margin:10px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.95em; font-weight:600; color:#ccc;'>⏳ Upcoming Restocks</div>", unsafe_allow_html=True)

# Countdown placeholders (embedded under Data Summary)
countdown_high = st.sidebar.empty()
countdown_medium = st.sidebar.empty()
countdown_low = st.sidebar.empty()

def render_countdown_block(confidence_label, color, placeholder):
    """Compact, balanced countdown block with smaller labels and larger numbers."""
    if consensus_summary.empty or confidence_label not in consensus_summary['Confidence'].values:
        placeholder.markdown(
            f"<div style='color:{color}; font-size:0.85em; margin-top:2px;'>No {confidence_label}-confidence restock scheduled.</div>",
            unsafe_allow_html=True
        )
        return

    next_event = consensus_summary[consensus_summary['Confidence'] == confidence_label].iloc[0]
    target_time = next_event['time_group']
    if target_time.tzinfo is None:
        target_time = pytz.timezone(selected_tz).localize(target_time)
    target_timestamp_ms = int(target_time.timestamp() * 1000)

    js = f"""
    <div style='margin-top:4px; margin-bottom:2px;'>
        <div style='color:{color}; font-weight:600; font-size:0.8em; margin-bottom:2px;'>
            Next {confidence_label} Confidence Restock:
        </div>
        <div id="countdown_{confidence_label}" 
             style="color:{color}; font-size:1.3em; font-weight:800; margin-left:2px; line-height:1.1;"></div>
    </div>
    <script>
    var targetTime_{confidence_label} = {target_timestamp_ms};
    function updateCountdown_{confidence_label}() {{
        var now = new Date().getTime();
        var diff = targetTime_{confidence_label} - now;
        if (diff <= 0) {{
            document.getElementById("countdown_{confidence_label}").innerHTML = "Now!";
            clearInterval(interval_{confidence_label});
            return;
        }}
        var d = Math.floor(diff / (1000*60*60*24));
        var h = Math.floor((diff % (1000*60*60*24)) / (1000*60*60));
        var m = Math.floor((diff % (1000*60*60)) / (1000*60));
        var s = Math.floor((diff % (1000*60)) / 1000);
        document.getElementById("countdown_{confidence_label}").innerHTML =
            d + "d " + h + "h " + m + "m " + s + "s";
    }}
    var interval_{confidence_label} = setInterval(updateCountdown_{confidence_label}, 1000);
    updateCountdown_{confidence_label}();
    </script>
    """
    with placeholder.container():
        components.html(js, height=50)








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
import pickle
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

@st.cache_resource
def load_pretrained_models(retailer):
    try:
        prophet_model = pickle.load(open(f"models/prophet_{retailer}.pkl", "rb"))
        lgbm_model = lgb.Booster(model_file=f"models/lgbm_{retailer}.txt")
        xgb_model = xgb.Booster(model_file=f"models/xgb_{retailer}.json")
        cat_model = cb.CatBoostRegressor()
        cat_model.load_model(f"models/cat_{retailer}.cbm")
        return prophet_model, lgbm_model, xgb_model, cat_model
    except Exception as e:
        st.error(f"Error loading pre-trained models for {retailer}: {e}")
        st.stop()

with st.spinner("Loading pre-trained models..."):
    prophet_model, lgbm_model, xgb_model, cat_model = load_pretrained_models(retailer)

    # -------------------------------------
    # 🔮 Generate Predictions Using Pre-Trained Models
    # -------------------------------------
    periods = forecast_horizon * 24 * 4  # 15-min intervals

    # Prophet Prediction
    future = prophet_model.make_future_dataframe(periods=periods, freq="15T")
    prophet_forecast_raw = prophet_model.predict(future)
    prophet_forecast_raw['ds'] = pd.to_datetime(prophet_forecast_raw['ds']).dt.tz_localize(pytz.UTC)
    prophet_forecast_raw['yhat'] = prophet_forecast_raw['yhat'].clip(lower=0)
    prophet_forecast_raw['Weekday'] = prophet_forecast_raw['ds'].dt.day_name()

    # -------------------------------------
    # ⚙️ Prepare Features for ML Models
    # -------------------------------------
    retailer_featured = create_features_for_ml(retailer_df)
    last_date = retailer_featured['DateTime'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=15), periods=periods, freq='15T')

    # Build future feature set
    future_df = pd.DataFrame({'DateTime': future_dates})
    full_history = pd.concat([retailer_featured, future_df], ignore_index=True)
    full_history = create_features_for_ml(full_history)

    FEATURES = [
        'hour', 'quarter', 'dayofweek', 'dayofyear', 'weekofyear', 'month',
        'tod_sin', 'tod_cos', 'dayofweek_sin', 'dayofweek_cos',
        'is_weekend', 'is_business_hours', 'lag_15m', 'lag_1h', 'lag_1d', 'lag_1w'
    ]
    X_future = full_history.iloc[-len(future_df):][FEATURES]

    # -------------------------------------
    # ⚡ Predict with LightGBM, XGBoost, CatBoost
    # -------------------------------------

    # LightGBM
    try:
        lgbm_yhat = np.clip(lgbm_model.predict(X_future), 0, None)
    except Exception:
        lgbm_yhat = np.clip(lgbm_model.predict(X_future.values), 0, None)

    # XGBoost
    dtest = xgb.DMatrix(X_future)
    xgb_yhat = np.clip(xgb_model.predict(dtest), 0, None)

    # CatBoost
    cat_yhat = np.clip(cat_model.predict(X_future), 0, None)

    # Helper function to create forecast DataFrame
    def make_forecast_df(future_dates, yhat):
        df = pd.DataFrame({'ds': future_dates, 'yhat': yhat})
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(pytz.UTC)
        df['Weekday'] = df['ds'].dt.day_name()
        return df

    lgbm_forecast_raw = make_forecast_df(future_dates, lgbm_yhat)
    xgb_forecast_raw = make_forecast_df(future_dates, xgb_yhat)
    cat_forecast_raw = make_forecast_df(future_dates, cat_yhat)




    FEATURES = ['hour', 'quarter', 'dayofweek', 'dayofyear', 'weekofyear', 'month',
                'tod_sin', 'tod_cos', 'dayofweek_sin', 'dayofweek_cos',
                'is_weekend', 'is_business_hours', 'lag_15m', 'lag_1h', 'lag_1d', 'lag_1w']
    try:
        # LightGBM importance (works for Booster)
        lgbm_imp = lgbm_model.feature_importance()

        # XGBoost importance
        xgb_gain = xgb_model.get_score(importance_type='gain')
        xgb_imp = np.array([xgb_gain.get(f, 0.0) for f in FEATURES])

        # CatBoost importance
        cat_imp = cat_model.get_feature_importance()

        # Normalize and average all three
        lgbm_imp = lgbm_imp / (lgbm_imp.sum() or 1)
        xgb_imp = xgb_imp / (xgb_imp.sum() or 1)
        cat_imp = cat_imp / (cat_imp.sum() or 1)
        avg_imp = (lgbm_imp + xgb_imp + cat_imp) / 3

        importance_df = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': avg_imp
        }).sort_values('Importance', ascending=False)

    except Exception:
        # fallback: equal weighting if any model doesn’t support importance lookup
        importance_df = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': [1 / len(FEATURES)] * len(FEATURES)
        }).sort_values('Importance', ascending=False)


# Filter only future predictions (in UTC) and convert to local for display
prophet_forecast = filter_by_time(prophet_forecast_raw)
xgb_forecast = filter_by_time(xgb_forecast_raw)
lgbm_forecast = filter_by_time(lgbm_forecast_raw)
cat_forecast = filter_by_time(cat_forecast_raw)

# Convert all forecast times to viewer's local timezone for display and charting
prophet_forecast = convert_to_local_time(prophet_forecast, time_col='ds', tz_str=selected_tz)
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
# --- Render Countdowns after consensus_summary exists ---


if consensus_summary.empty:
    countdown_high.markdown("<span style='color:#999;'>No predictions available.</span>", unsafe_allow_html=True)
    countdown_medium.empty()
    countdown_low.empty()
else:
    render_countdown_block("High", "#28a745", countdown_high)
    render_countdown_block("Medium", "#FFA500", countdown_medium)
    render_countdown_block("Low", "#FF4B4B", countdown_low)


# -------------------------
# 7. Main Tabs
# -------------------------
tab_names = ["⭐ Prediction Schedule", "📊 Analysis", "🔮 Prophet", "🚀 LightGBM", "🌟 XGBoost", "🐾 CatBoost"]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.header("Prediction Schedule")
    consensus_fig = create_consensus_chart(consensus_summary)
    st.plotly_chart(consensus_fig, use_container_width=True)
    display_consensus_schedule(consensus_summary)


with tabs[1]:
    st.header("Analysis Heatmaps")
    st.markdown("These charts show the typical patterns of restock activity, based on model forecasts.")
    st.subheader("Hourly Activity Heatmap")
    if prophet_forecast is not None and not prophet_forecast.empty:
        hourly_fig = create_hourly_heatmap(prophet_forecast, retailer)
        st.plotly_chart(hourly_fig, use_container_width=True)
    st.subheader("Model Feature Importance")
    fig_imp = create_importance_chart(importance_df)
    st.plotly_chart(fig_imp, use_container_width=True)

with tabs[2]:
    st.header("🔮 Prophet Model Details")
    fig_prophet = create_forecast_chart(prophet_forecast, prophet_big, retailer, "Prophet Forecast", chart_color)
    st.plotly_chart(fig_prophet, use_container_width=True)
with tabs[3]:
    st.header("🚀 LightGBM Model Details")
    fig_lgbm = create_forecast_chart(lgbm_forecast, lgbm_big, retailer, "LightGBM Forecast", chart_color)
    st.plotly_chart(fig_lgbm, use_container_width=True)
with tabs[4]:
    st.header("🌟 XGBoost Model Details")
    fig_xgb = create_forecast_chart(xgb_forecast, xgb_big, retailer, "XGBoost Forecast", chart_color)
    st.plotly_chart(fig_xgb, use_container_width=True)
with tabs[5]:
    st.header("🐾 CatBoost Model Details")
    fig_cat = create_forecast_chart(cat_forecast, cat_big, retailer, "CatBoost Forecast", chart_color)
    st.plotly_chart(fig_cat, use_container_width=True)
