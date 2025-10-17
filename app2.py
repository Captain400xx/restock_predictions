# ----------------------------------------------------------------------
# Pokémon Card Drop Forecast - v5.6 (Chart Readability Tweak)
# ----------------------------------------------------------------------

# -------------------------
# 1. Import Libraries
# -------------------------
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
import pytz
from io import StringIO
import warnings
from datetime import datetime
import numpy as np
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# -------------------------
# 2. App Configuration & Initial Setup
# -------------------------
st.set_page_config(page_title="Pokémon Restock Forecast", layout="wide", initial_sidebar_state="expanded")
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
            return df.to_csv(index=False)
        except Exception as e:
            st.error(f"Error reading your CSV file: {e}")
            return None
    else:
        st.warning(f"File '{file_path}' not found. Using sample data. To use your own data, save it as {file_path} in the same folder.")
        default_csv_data = """DateTime,Retailer
        2025-09-01 10:05,Pokemon Center
        2025-09-02 14:10,Walmart
        2025-09-03 11:30,Target
        2025-09-04 18:01,Pokemon Center
        2025-09-08 10:01,Pokemon Center
        2025-09-09 14:05,Walmart
        2025-09-15 10:05,Pokemon Center
        2025-09-16 14:15,Walmart
        2025-09-18 18:00,Pokemon Center
        2025-09-22 10:02,Pokemon Center
        2025-09-23 14:00,Walmart
        2025-09-29 10:00,Pokemon Center
        2025-09-30 14:10,Walmart
        2025-10-02 18:00,Pokemon Center
        2025-10-06 10:05,Pokemon Center
        2025-10-07 14:00,Walmart
        2025-10-13 10:01,Pokemon Center
        2025-10-14 14:20,Walmart
        2025-10-16 18:00,Pokemon Center
        """
        return default_csv_data

# -------------------------
# 4. Helper Functions
# -------------------------
@st.cache_data
def process_data_for_forecasting(csv_data_string):
    df = pd.read_csv(StringIO(csv_data_string))
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Count'] = 1
    df_hourly = df.set_index('DateTime').groupby('Retailer').resample('H').sum(numeric_only=True).drop(columns='Retailer', errors='ignore').reset_index()
    all_retailers = df_hourly["Retailer"].unique()
    filled_dfs = []
    min_date, max_date = df_hourly['DateTime'].min(), df_hourly['DateTime'].max()
    for r in all_retailers:
        retailer_df = df_hourly[df_hourly["Retailer"] == r].set_index("DateTime")
        full_range = pd.date_range(start=min_date, end=max_date, freq='H')
        retailer_df = retailer_df.reindex(full_range, fill_value=0)
        retailer_df['Retailer'] = r
        filled_dfs.append(retailer_df.reset_index().rename(columns={'index': 'DateTime'}))
    processed_df = pd.concat(filled_dfs, ignore_index=True)
    return processed_df

def filter_by_time(df):
    now_eastern = datetime.now(EASTERN)
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    return df[df['ds'] >= now_eastern.replace(tzinfo=None)]

def create_time_features(df):
    df['hour'] = df['DateTime'].dt.hour
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['quarter'] = df['DateTime'].dt.quarter
    df['month'] = df['DateTime'].dt.month
    df['year'] = df['DateTime'].dt.year
    df['dayofyear'] = df['DateTime'].dt.dayofyear
    df['weekofyear'] = df['DateTime'].dt.isocalendar().week.astype(int)
    return df

# ----- MODEL TRAINING FUNCTIONS -----
@st.cache_data
def train_prophet_model(data, forecast_horizon, ci_width):
    df_prophet = data.rename(columns={"DateTime": "ds", "Count": "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False, interval_width=ci_width / 100.0)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_horizon, freq="H")
    forecast = model.predict(future)
    forecast['Weekday'] = forecast['ds'].dt.day_name()
    return model, forecast

@st.cache_data
def train_ml_model(data, forecast_horizon, model_type='lgbm'):
    df_featured = create_time_features(data)
    FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'weekofyear']
    TARGET = 'Count'

    X_train = df_featured[FEATURES]
    y_train = df_featured[TARGET]

    if model_type == 'lgbm':
        model = lgb.LGBMRegressor(random_state=42)
    else:
        model = xgb.XGBRegressor(random_state=42)
    
    model.fit(X_train, y_train)

    last_date = data['DateTime'].max()
    future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='H', inclusive='right')
    future_df = pd.DataFrame({'DateTime': future_dates})
    future_df_featured = create_time_features(future_df)
    X_future = future_df_featured[FEATURES]
    future_df['yhat'] = model.predict(X_future)
    
    future_df['yhat_lower'] = future_df['yhat'] - (future_df['yhat'] * 0.5)
    future_df['yhat_upper'] = future_df['yhat'] + (future_df['yhat'] * 0.5)

    forecast_df = future_df.rename(columns={'DateTime': 'ds'})
    forecast_df['Weekday'] = forecast_df['ds'].dt.day_name()
    
    return model, forecast_df

# ----- ANALYSIS & PLOTTING -----
def get_big_restocks(forecast_df, method="percentile", threshold=90, abs_value=None):
    forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0)
    if method == "percentile":
        cutoff = forecast_df["yhat"].quantile(threshold / 100.0)
    else:
        cutoff = float(abs_value) if abs_value is not None else 0
    cutoff = max(cutoff, 0.5) 
    big = forecast_df[forecast_df["yhat"] >= cutoff].copy()
    return cutoff, big.sort_values("ds")

def create_forecast_chart(forecast_df, big_restocks_df, historical_df, retailer, title, show_history=False):
    fig = go.Figure()
    if show_history:
        fig.add_trace(go.Scatter(x=historical_df["DateTime"], y=historical_df["Count"], mode='markers', name='Historical Activity', marker=dict(color='grey', opacity=0.6, size=5), hovertemplate='%{x|%Y-%m-%d %H:%M} — %{y:.0f} events<extra></extra>'))
    
    fig.add_trace(go.Scatter(x=list(forecast_df["ds"]) + list(forecast_df["ds"][::-1]), y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"][::-1]), fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Confidence Interval"))
    
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="Forecast", line=dict(width=2, color="orange"), hovertemplate='%{x|%Y-%m-%d %H:%M} — %{y:.1f} activity<extra></extra>'))
    
    if not big_restocks_df.empty:
        fig.add_trace(go.Scatter(x=big_restocks_df["ds"], y=big_restocks_df["yhat"], mode="markers", name="Predicted Big Restock", marker=dict(size=10, color='red', symbol='star'), hovertemplate='%{x|%Y-%m-%d %H:%M} — %{y:.1f} activity<extra></extra>'))
    
    now_eastern = datetime.now(EASTERN)
    
    # --- CHART LAYOUT IMPROVEMENTS ---
    # 1. Dynamic Y-Axis Zoom
    # Calculate a sensible y-axis max based on the data, not the confidence interval
    if not forecast_df.empty:
        ymax = max(forecast_df['yhat'].max(), forecast_df['yhat_upper'].quantile(0.95)) # Use 95th percentile of CI
        yaxis_max = ymax * 1.2 # Add 20% buffer
    else:
        yaxis_max = 10 # Default if no data

    fig.update_layout(
        title=f"{title} for {retailer}",
        xaxis_title="Date & Time (US/Eastern)",
        yaxis_title="Predicted Restock Activity",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        yaxis_range=[-1, yaxis_max], # Apply dynamic zoom
        xaxis=dict(
            tickformat="%a %m/%d", # Universal format for date
            dtick="D1",
            minor=dict(
                dtick=60 * 60 * 6 * 1000, # 6 hours in milliseconds
                tickmode="auto",
                showgrid=True,
                # 2. More visible gridlines
                gridcolor='rgba(0, 0, 0, 0.1)' # Made darker
            )
        ),
        shapes=[dict(type='line', x0=now_eastern, y0=0, x1=now_eastern, y1=1, yref='paper', line=dict(color='RoyalBlue', width=2, dash='dash'))],
        annotations=[dict(x=now_eastern, y=1.05, yref='paper', text='Current Time', showarrow=False)]
    )
    return fig


def create_calendar_heatmap(forecast_df, retailer):
    df = forecast_df.copy()
    df['date'] = df['ds'].dt.date
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
        except IndexError:
            continue
    fig = go.Figure(data=go.Heatmap(z=heatmap_data, x=[f"Week {w}" for w in weeks], y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], hoverongaps=False, text=text_data, texttemplate="%{text}", colorscale='Oranges', colorbar_title='Max Activity'))
    fig.update_layout(title=f"Calendar View: Daily Max Predicted Activity for {retailer}", xaxis_title="Week of the Year", height=400)
    return fig

# -------------------------
# 5. Main App Logic
# -------------------------
st.title("🃏 Pokémon Card Restock Forecast")

raw_data_string = load_main_data(DATA_FILE)

if raw_data_string is None:
    st.stop()

full_df = process_data_for_forecasting(raw_data_string)

with st.expander("👋 How to Use This App", expanded=True):
    st.markdown("""
    This app forecasts Pokémon card restock activity using three different models.

    - **Select a Retailer:** Use the dropdown in the sidebar to choose a retailer.
    - **Set the Forecast Horizon:** Use the slider to decide how far into the future to predict.
    - **Define a 'Big Restock':** Choose how to identify significant predictions.
    - **Explore the Tabs:**
        - **⭐ Prediction Consensus:** See a summary of high-confidence predictions where multiple models agree.
        - **📅 Calendar View:** Get a visual overview of the hottest days.
        - **Model Tabs (Prophet, LightGBM, XGBoost):** Dive deep into the forecast of each individual model.

    **What is 'Predicted Restock Activity'?** Since we are counting every restock event as '1', the model forecasts the **frequency of restock events per hour**.
    - A **low** score (e.g., 0-1) means no or very few restock events are expected.
    - A **high** score (e.g., 5+) means the model predicts a cluster of many separate restock events in that hour.
    """)

# -------------------------
# 6. Sidebar Controls
# -------------------------
st.sidebar.title("⚙️ Controls")

retailers = sorted(full_df["Retailer"].unique().tolist())
if not retailers:
    st.error("No retailers found in the data. Please check your CSV file.")
    st.stop()

default_index = 0
if 'Target' in retailers:
    default_index = retailers.index('Target')

retailer = st.sidebar.selectbox("Choose a retailer to forecast", retailers, index=default_index)

st.sidebar.markdown("---")
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 30, 14, 1)
forecast_hours = forecast_horizon * 24
st.sidebar.subheader("Big Restock Definition")
restock_method = st.sidebar.selectbox("Detection Method", ["percentile", "absolute"])
if restock_method == "percentile":
    percentile_val = st.sidebar.slider("Percentile Threshold", 50, 99, 90)
    abs_val = None
else:
    abs_val = st.sidebar.number_input("Absolute Threshold (Activity ≥)", 1, value=5)
    percentile_val = None
st.sidebar.markdown("---")
st.sidebar.subheader("Prophet Model Tuning")
prophet_ci = st.sidebar.slider("Prophet Confidence Interval (%)", 70, 99, 80)

retailer_df = full_df[full_df["Retailer"] == retailer].copy()

# --- Run All Models & Generate Consensus ---
with st.spinner("Training models and generating forecasts... this may take a moment."):
    prophet_model, prophet_forecast_raw = train_prophet_model(retailer_df, forecast_hours, prophet_ci)
    xgb_model, xgb_forecast_raw = train_ml_model(retailer_df, forecast_hours, model_type='xgb')
    lgbm_model, lgbm_forecast_raw = train_ml_model(retailer_df, forecast_hours, model_type='lgbm')

prophet_forecast = filter_by_time(prophet_forecast_raw)
xgb_forecast = filter_by_time(xgb_forecast_raw)
lgbm_forecast = filter_by_time(lgbm_forecast_raw)

_, prophet_big = get_big_restocks(prophet_forecast, restock_method, percentile_val, abs_val)
_, xgb_big = get_big_restocks(xgb_forecast, restock_method, percentile_val, abs_val)
_, lgbm_big = get_big_restocks(lgbm_forecast, restock_method, percentile_val, abs_val)

all_big_restocks = []
for _, row in prophet_big.iterrows(): all_big_restocks.append({'ds': row['ds'], 'model': 'Prophet', 'yhat': row['yhat']})
for _, row in xgb_big.iterrows(): all_big_restocks.append({'ds': row['ds'], 'model': 'XGBoost', 'yhat': row['yhat']})
for _, row in lgbm_big.iterrows(): all_big_restocks.append({'ds': row['ds'], 'model': 'LightGBM', 'yhat': row['yhat']})

if all_big_restocks:
    consensus_df = pd.DataFrame(all_big_restocks).sort_values('ds').reset_index(drop=True)
    consensus_df['time_group'] = consensus_df['ds'].dt.round('3H')
    consensus_summary = consensus_df.groupby('time_group').agg(models=('model', lambda x: ', '.join(sorted(x.unique()))), model_count=('model', 'nunique'), avg_activity=('yhat', 'mean')).reset_index().sort_values('model_count', ascending=False)
    def get_confidence(count):
        if count >= 3: return "High"
        if count == 2: return "Medium"
        return "Low"
    consensus_summary['Confidence'] = consensus_summary['model_count'].apply(get_confidence)
    consensus_summary = consensus_summary.sort_values(['model_count', 'time_group'], ascending=[False, True])
else:
    consensus_summary = pd.DataFrame()

# --- Top-Level Summary ---
st.markdown("---")
st.subheader("🚀 Top Prediction")
if not consensus_summary.empty and "High" in consensus_summary['Confidence'].values:
    top_pred = consensus_summary[consensus_summary['Confidence'] == 'High'].iloc[0]
    
    time_str = top_pred['time_group'].strftime('%A, %b %d at %I %p')
    
    st.success(f"**Next High-Confidence Restock Window:** Around **{time_str}**\n- **Models in Agreement:** {top_pred['models']}\n- **Average Predicted Activity:** {top_pred['avg_activity']:.1f}")
else:
    st.info("No 'High Confidence' restocks (where all 3 models agree) were found in the forecast horizon.")

# -------------------------
# 7. Main Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["⭐ Prediction Consensus", "📅 Calendar View", "🔮 Prophet", "🚀 LightGBM", "🌟 XGBoost"])
with tab1:
    st.header("Prediction Consensus")
    st.markdown("This table shows predicted 'big restock' events. **Confidence** is higher when more models agree.")
    if not consensus_summary.empty:
        display_consensus = consensus_summary.rename(columns={'time_group': 'Approx. Date & Time', 'models': 'Models in Agreement', 'avg_activity': 'Avg. Predicted Activity'})[['Confidence', 'Approx. Date & Time', 'Avg. Predicted Activity', 'Models in Agreement']]
        st.dataframe(display_consensus, use_container_width=True, hide_index=True)
    else:
        st.warning("No significant restock activity was predicted based on the current settings.")
with tab2:
    st.header("Calendar Heatmap")
    st.markdown("This calendar shows the max predicted activity for each day, based on **Prophet's forecast**.")
    if not prophet_forecast.empty:
        calendar_fig = create_calendar_heatmap(prophet_forecast, retailer)
        st.plotly_chart(calendar_fig, use_container_width=True)
with tab3:
    st.header("🔮 Prophet Model Details")
    show_history_prophet = st.checkbox("Show Historical Data", key="prophet_hist")
    fig_prophet = create_forecast_chart(prophet_forecast, prophet_big, retailer_df, retailer, "Prophet Forecast", show_history_prophet)
    st.plotly_chart(fig_prophet, use_container_width=True)
    st.subheader("Upcoming Big Restocks (Prophet)")
    if prophet_big.empty: st.write("None predicted.")
    else: st.dataframe(prophet_big.rename(columns={'ds': 'Date & Time', 'yhat': 'Predicted Activity', 'yhat_lower': 'Predicted Low', 'yhat_upper': 'Predicted High'})[['Date & Time', 'Predicted Activity', 'Predicted Low', 'Predicted High']], use_container_width=True, hide_index=True)
    st.subheader("Prophet Model Components")
    fig_components = prophet_model.plot_components(prophet_forecast_raw)
    st.pyplot(fig_components)
with tab4:
    st.header("🚀 LightGBM Model Details")
    show_history_lgbm = st.checkbox("Show Historical Data", key="lgbm_hist")
    fig_lgbm = create_forecast_chart(lgbm_forecast, lgbm_big, retailer_df, retailer, "LightGBM Forecast", show_history_lgbm)
    st.plotly_chart(fig_lgbm, use_container_width=True)
    st.subheader("Upcoming Big Restocks (LightGBM)")
    if lgbm_big.empty: st.write("None predicted.")
    else: st.dataframe(lgbm_big.rename(columns={'ds': 'Date & Time', 'yhat': 'Predicted Activity', 'yhat_lower': 'Predicted Low', 'yhat_upper': 'Predicted High'})[['Date & Time', 'Predicted Activity', 'Predicted Low', 'Predicted High']], use_container_width=True, hide_index=True)
with tab5:
    st.header("🌟 XGBoost Model Details")
    show_history_xgb = st.checkbox("Show Historical Data", key="xgb_hist")
    fig_xgb = create_forecast_chart(xgb_forecast, xgb_big, retailer_df, retailer, "XGBoost Forecast", show_history_xgb)
    st.plotly_chart(fig_xgb, use_container_width=True)
    st.subheader("Upcoming Big Restocks (XGBoost)")
    if xgb_big.empty: st.write("None predicted.")
    else: st.dataframe(xgb_big.rename(columns={'ds': 'Date & Time', 'yhat': 'Predicted Activity', 'yhat_lower': 'Predicted Low', 'yhat_upper': 'Predicted High'})[['Date & Time', 'Predicted Activity', 'Predicted Low', 'Predicted High']], use_container_width=True, hide_index=True)