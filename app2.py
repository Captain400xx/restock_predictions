# ----------------------------------------------------------------------
# Pokémon Card Drop Forecast - v7.5 (Ensure forecast_hours definition)
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
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')
    df['hour'] = df['DateTime'].dt.hour
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['dayofyear'] = df['DateTime'].dt.dayofyear
    df['weekofyear'] = df['DateTime'].dt.isocalendar().week.astype(int)
    df['month'] = df['DateTime'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    df['lag_1h'] = df['Count'].shift(1).fillna(0)
    df['lag_24h'] = df['Count'].shift(24).fillna(0)
    df['lag_1w'] = df['Count'].shift(24 * 7).fillna(0)
    return df

@st.cache_data
def process_data_for_forecasting(csv_data_string):
    df = pd.read_csv(StringIO(csv_data_string))
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Count'] = 1
    df_hourly = df.set_index('DateTime').groupby('Retailer').resample('H').sum(numeric_only=True).drop(columns='Retailer', errors='ignore').reset_index()
    all_retailers = df_hourly["Retailer"].unique()
    processed_dfs = []
    for r in all_retailers:
        retailer_df = df_hourly[df_hourly["Retailer"] == r].copy()
        full_range = pd.date_range(start=retailer_df['DateTime'].min(), end=retailer_df['DateTime'].max(), freq='H')
        retailer_df = retailer_df.set_index('DateTime').reindex(full_range).fillna(0).reset_index().rename(columns={'index': 'DateTime'})
        retailer_df['Retailer'] = r
        retailer_featured_df = create_features_for_ml(retailer_df)
        processed_dfs.append(retailer_featured_df)
    return pd.concat(processed_dfs, ignore_index=True)


def filter_by_time(df, future_only=True):
    now_eastern = datetime.now(EASTERN)
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    if future_only:
        return df[df['ds'] >= now_eastern.replace(tzinfo=None)]
    else:
        return df

# ----- MODEL TRAINING FUNCTIONS -----
@st.cache_data
def train_prophet_model(data, forecast_horizon):
    df_prophet = data.rename(columns={"DateTime": "ds", "Count": "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_horizon, freq="H")
    forecast = model.predict(future)
    forecast['Weekday'] = forecast['ds'].dt.day_name()
    return model, forecast

@st.cache_data
def train_ml_model(data, forecast_horizon, model_type='lgbm'):
    FEATURES = ['hour', 'dayofweek', 'dayofyear', 'weekofyear', 'month','hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos','is_weekend', 'is_business_hours', 'lag_1h', 'lag_24h', 'lag_1w']
    TARGET = 'Count'
    X_train = data[FEATURES]
    y_train = data[TARGET]
    if model_type == 'lgbm': model = lgb.LGBMRegressor(random_state=42, objective='poisson')
    elif model_type == 'xgb': model = xgb.XGBRegressor(random_state=42, objective='count:poisson')
    else: model = cb.CatBoostRegressor(random_state=42, loss_function='Poisson', verbose=0)
    model.fit(X_train, y_train)
    last_date = data['DateTime'].max()
    future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='H', inclusive='right')
    future_df = pd.DataFrame({'DateTime': future_dates})
    full_history = pd.concat([data.set_index('DateTime'), future_df.set_index('DateTime')]).reset_index()
    full_history_featured = create_features_for_ml(full_history.rename(columns={'index':'DateTime'}))
    X_future = full_history_featured.iloc[-len(future_df):][FEATURES]
    future_df['yhat'] = model.predict(X_future)
    forecast_df = future_df.rename(columns={'DateTime': 'ds'})
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    forecast_df['Weekday'] = forecast_df['ds'].dt.day_name()
    return model, forecast_df

# ----- ANALYSIS & PLOTTING -----
def get_big_restocks(forecast_df):
    forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0)
    cutoff = forecast_df["yhat"].quantile(0.90)
    cutoff = max(cutoff, 0.5) 
    big = forecast_df[forecast_df["yhat"] >= cutoff].copy()
    return cutoff, big.sort_values("ds")

def create_forecast_chart(forecast_df, big_restocks_df, retailer, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="Forecast", line=dict(width=2, color=color), hovertemplate='%{x|%Y-%m-%d %H:%M} — %{y:.1f} activity<extra></extra>'))
    if not big_restocks_df.empty:
        fig.add_trace(go.Scatter(x=big_restocks_df["ds"], y=big_restocks_df["yhat"], mode="markers", name="Predicted Big Restock", marker=dict(size=10, color='red', symbol='star'), hovertemplate='%{x|%Y-%m-%d %H:%M} — %{y:.1f} activity<extra></extra>'))
    now_eastern = datetime.now(EASTERN)
    ymax = max(forecast_df['yhat'].max() * 1.2, 5) if not forecast_df.empty else 5
    fig.update_layout(title=f"{title} for {retailer}",hovermode="x unified",template="plotly_white",height=500,yaxis=dict(title="Predicted Restock Activity", range=[0, ymax]),xaxis=dict(title="Date (US/Eastern)", tickformat="%a %m/%d", dtick="D1", showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)'),xaxis2=dict(title="Time of Day (ET)", overlaying='x', side='top', tickformat='%I %p', dtick=3600000 * 4, showgrid=True, gridcolor='rgba(255, 0, 0, 0.3)', gridwidth=1, showticklabels=True),shapes=[dict(type='line', x0=now_eastern, y0=0, x1=now_eastern, y1=1, yref='paper', line=dict(color='RoyalBlue', width=2, dash='dash'))],annotations=[dict(x=now_eastern, y=1.05, yref='paper', text='Current Time', showarrow=False)])
    return fig

def display_consensus_schedule(df):
    st.markdown("""<style>.schedule-table { width: 100%; border-collapse: collapse; font-size: 0.9em; } .schedule-table th, .schedule-table td { padding: 8px; text-align: left; border-bottom: 1px solid #444; } .schedule-table th { background-color: #1a1a1a; } .high-confidence { background-color: rgba(40, 167, 69, 0.3); } .medium-confidence { background-color: rgba(255, 193, 7, 0.3); } .low-confidence { background-color: rgba(220, 53, 69, 0.3); } .day-group { font-weight: bold; font-size: 1.1em; padding-top: 15px; }</style>""", unsafe_allow_html=True)
    html = "<table class='schedule-table'>"
    html += "<tr><th>Time (ET)</th><th>Avg. Activity</th><th>Confidence</th><th>Models in Agreement</th></tr>"
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
    df['color'] = df['Confidence'].map(color_map)
    df['day_label'] = df['time_group'].dt.strftime('%a %m/%d')
    fig = go.Figure(go.Bar(x=df['time_group'],y=df['avg_activity'],marker_color=df['color'],hovertemplate="<b>%{x|%a, %b %d, %I %p}</b><br>Confidence: %{customdata[0]}<br>Avg. Activity: %{y:.1f}<extra></extra>",customdata=df[['Confidence']]))
    fig.update_layout(title="Visual Prediction Schedule",xaxis_title="Date & Time (ET)",yaxis_title="Average Predicted Activity",template="plotly_white",height=400,xaxis=dict(type='category',tickmode='array',tickvals=df['time_group'],ticktext=df['day_label']))
    return fig
    
def create_importance_chart(df):
    fig = go.Figure(go.Bar(x=df['Importance'],y=df['Feature'],orientation='h'))
    fig.update_layout(title="Model Feature Importance (All Models)",xaxis_title="Average Importance",yaxis_title="Feature",template="plotly_white",yaxis=dict(autorange="reversed"),height=400)
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
        except (IndexError, ValueError):
            continue
    fig = go.Figure(data=go.Heatmap(z=heatmap_data, x=[f"Week {w}" for w in weeks], y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], hoverongaps=False, text=text_data, texttemplate="%{text}", colorscale='Oranges', colorbar_title='Max Activity'))
    fig.update_layout(title=f"Daily Activity Heatmap for {retailer}", xaxis_title="Week of the Year", height=400)
    return fig

def create_hourly_heatmap(forecast_df, retailer):
    df = forecast_df.copy()
    df['hour'] = df['ds'].dt.hour
    df['dayofweek'] = df['ds'].dt.dayofweek
    hourly_avg = df.groupby(['dayofweek', 'hour'])['yhat'].mean().reset_index()
    heatmap_data = hourly_avg.pivot(index='dayofweek', columns='hour', values='yhat')
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hour_labels = [f"{h}:00" for h in range(24)]
    fig = go.Figure(data=go.Heatmap(z=heatmap_data.values,x=hour_labels,y=day_labels,colorscale='Oranges',colorbar_title='Avg. Activity'))
    fig.update_layout(title=f"Hourly Activity Heatmap for {retailer}", xaxis_title="Hour of Day (ET)", yaxis_title="Day of Week", height=500)
    return fig

# -------------------------
# 5. Main App Logic
# -------------------------
st.title("🃏 Pokémon Card Restock Forecast")
raw_data_string = load_main_data(DATA_FILE)
if raw_data_string is None: st.stop()
full_df = process_data_for_forecasting(raw_data_string)

with st.expander("👋 How to Use This App", expanded=True):
    st.markdown("""
    This app forecasts Pokémon card restock activity using four different models.
    - **Select a Retailer:** Use the dropdown in the sidebar.
    - **Set the Forecast Horizon:** Use the slider to decide how far into the future to predict.
    - **Explore the Tabs:**
        - **⭐ Prediction Schedule:** A visual schedule of high-confidence predictions.
        - **📅 2-Week View:** A chart to check model accuracy against the past week's data.
        - **📊 Analysis:** Heatmaps showing the hottest days and times for restocks.
        - **Model Tabs (Prophet, etc.):** Dive deep into the forecast of each individual model.

    **What is 'Predicted Restock Activity'?** Since we are counting every restock event as '1', the model forecasts the **frequency of restock events per hour**.
    - A **low** score (e.g., 0-1) means no or very few restock events are expected.
    - A **high** score (e.g., 5+) means the model predicts a cluster of many separate restock events in that hour.
    """)

# -------------------------
# 6. Sidebar Controls
# -------------------------
st.markdown("""<style>[data-testid="stSidebar"] > div:first-child {padding-top: 1rem;}</style>""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Controls")
retailers = sorted(full_df["Retailer"].unique().tolist())
if not retailers: st.error("No retailers found."); st.stop()

default_index = 0
if 'Target' in retailers: default_index = retailers.index('Target')
retailer = st.sidebar.selectbox("Choose a retailer to forecast", retailers, index=default_index, label_visibility="collapsed")

st.sidebar.markdown("---")
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 30, 14, 1)
# --- Ensure forecast_hours is defined here ---
forecast_hours = forecast_horizon * 24 

st.sidebar.markdown("---")
st.sidebar.subheader("Data Summary")
retailer_history_df = full_df[full_df['Retailer'] == retailer]
min_date = retailer_history_df['DateTime'].min().strftime('%b %d, %Y')
max_date = retailer_history_df['DateTime'].max().strftime('%b %d, %Y')
event_count = int(retailer_history_df['Count'].sum())
st.sidebar.markdown(f"""<div style="font-size: 0.9em;">Data Available From: <strong style="font-size: 1.1em;">{min_date}</strong><br>Data Available To: <strong style="font-size: 1.1em;">{max_date}</strong><br>Total Recorded Events: <strong style="font-size: 1.1em;">{event_count:,}</strong></div>""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Chart Customization")
chart_color = st.sidebar.color_picker("Forecast Line Color", "#FFA500")

st.sidebar.markdown("**Next High-Confidence Alert:**")
countdown_placeholder = st.sidebar.empty()
st.sidebar.markdown("---")
if os.path.exists("logo.png"): st.sidebar.image("logo.png", width=100)
st.sidebar.warning("This data is the property of RestockR Monitors. Unauthorized sharing, distribution, or external use of this information may result in penalties or legal action.")
st.sidebar.markdown("---")
st.sidebar.markdown("App by **Captain400x**")

retailer_df = full_df[full_df["Retailer"] == retailer].copy()

# --- Run All Models & Generate Consensus ---
with st.spinner("Training models with advanced features... this may take a moment."):
    prophet_model, prophet_forecast_raw = train_prophet_model(retailer_df, forecast_hours)
    xgb_model, xgb_forecast_raw = train_ml_model(retailer_df, forecast_hours, model_type='xgb')
    lgbm_model, lgbm_forecast_raw = train_ml_model(retailer_df, forecast_hours, model_type='lgbm')
    cat_model, cat_forecast_raw = train_ml_model(retailer_df, forecast_hours, model_type='catboost')
    
    FEATURES = ['hour', 'dayofweek', 'dayofyear', 'weekofyear', 'month','hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos','is_weekend', 'is_business_hours', 'lag_1h', 'lag_24h', 'lag_1w']
    lgbm_imp = lgbm_model.feature_importances_
    xgb_imp = xgb_model.feature_importances_
    cat_imp = cat_model.feature_importances_
    avg_imp = (lgbm_imp/lgbm_imp.sum() + xgb_imp/xgb_imp.sum() + cat_imp/cat_imp.sum()) / 3
    importance_df = pd.DataFrame({'Feature': FEATURES, 'Importance': avg_imp}).sort_values('Importance', ascending=False)


prophet_forecast = filter_by_time(prophet_forecast_raw)
xgb_forecast = filter_by_time(xgb_forecast_raw)
lgbm_forecast = filter_by_time(lgbm_forecast_raw)
cat_forecast = filter_by_time(cat_forecast_raw)

_, prophet_big = get_big_restocks(prophet_forecast)
_, xgb_big = get_big_restocks(xgb_forecast)
_, lgbm_big = get_big_restocks(lgbm_forecast)
_, cat_big = get_big_restocks(cat_forecast)

all_big_restocks = []
for _, row in prophet_big.iterrows(): all_big_restocks.append({'ds': row['ds'], 'model': 'Prophet', 'yhat': row['yhat']})
for _, row in xgb_big.iterrows(): all_big_restocks.append({'ds': row['ds'], 'model': 'XGBoost', 'yhat': row['yhat']})
for _, row in lgbm_big.iterrows(): all_big_restocks.append({'ds': row['ds'], 'model': 'LightGBM', 'yhat': row['yhat']})
for _, row in cat_big.iterrows(): all_big_restocks.append({'ds': row['ds'], 'model': 'CatBoost', 'yhat': row['yhat']})

if all_big_restocks:
    consensus_df = pd.DataFrame(all_big_restocks).sort_values('ds').reset_index(drop=True)
    consensus_df['time_group'] = consensus_df['ds'].dt.round('3H')
    consensus_summary = consensus_df.groupby('time_group').agg(models=('model', lambda x: ', '.join(sorted(x.unique()))), model_count=('model', 'nunique'), avg_activity=('yhat', 'mean')).reset_index()
    def get_confidence(count):
        if count >= 3: return "High"
        if count == 2: return "Medium"
        return "Low"
    consensus_summary['Confidence'] = consensus_summary['model_count'].apply(get_confidence)
    consensus_summary = consensus_summary.sort_values('time_group', ascending=True)
else:
    consensus_summary = pd.DataFrame()

# --- Top Prediction & Countdown ---
st.markdown("---")
st.subheader("🚀 Top Prediction")
if not consensus_summary.empty and "High" in consensus_summary['Confidence'].values:
    top_pred = consensus_summary[consensus_summary['Confidence'] == 'High'].iloc[0]
    time_str = top_pred['time_group'].strftime('%A, %b %d at %I %p')
    
    st.success(f"**Next High-Confidence Restock Window:** Around **{time_str}**\n- **Models in Agreement:** {top_pred['models']}\n- **Average Predicted Activity:** {top_pred['avg_activity']:.1f}")
    
    now_eastern = datetime.now(EASTERN)
    target_time_naive = top_pred['time_group']
    if target_time_naive.tzinfo is None:
        target_time_aware = EASTERN.localize(target_time_naive)
    else:
        target_time_aware = target_time_naive.astimezone(EASTERN)
    target_timestamp_ms = int(target_time_aware.timestamp() * 1000)
    
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
    st.markdown("This chart shows how well the **Prophet model's forecast (line)** matched the **actual historical data (line)** for the past 7 days, giving you a guide for how to interpret the next 7 days.")
    now = datetime.now(EASTERN)
    start_date = now - timedelta(days=7)
    end_date = now + timedelta(days=7)
    history_in_window = retailer_history_df[(retailer_history_df['DateTime'] >= start_date.replace(tzinfo=None)) & (retailer_history_df['DateTime'] <= now.replace(tzinfo=None))]
    prophet_full_forecast = filter_by_time(prophet_forecast_raw, future_only=False)
    forecast_in_window = prophet_full_forecast[(prophet_full_forecast['ds'] >= start_date.replace(tzinfo=None)) & (prophet_full_forecast['ds'] <= end_date.replace(tzinfo=None))]
    fig_2_week = go.Figure()
    fig_2_week.add_trace(go.Scatter(x=history_in_window['DateTime'], y=history_in_window['Count'], mode='lines', name='Actual Past Activity', line=dict(color='blue', width=2)))
    fig_2_week.add_trace(go.Scatter(x=forecast_in_window['ds'], y=forecast_in_window['yhat'], mode='lines', name='Forecast', line=dict(color=chart_color, width=3, dash='dot')))
    fig_2_week.add_shape(type="line",x0=now.replace(tzinfo=None), x1=now.replace(tzinfo=None),y0=0, y1=1,yref="paper",line=dict(color="red", width=3, dash="dash"))
    fig_2_week.add_annotation(x=now.replace(tzinfo=None), y=1.05, yref="paper", text="Current Time", showarrow=False)
    fig_2_week.update_layout(title=f"Past Week vs. Next Week Forecast for {retailer}", xaxis_title="Date", yaxis_title="Restock Activity", template="plotly_white", height=500)
    st.plotly_chart(fig_2_week, use_container_width=True)

with tabs[2]:
    st.header("Analysis Heatmaps")
    st.markdown("These charts show the typical patterns of restock activity, based on the **Prophet model's forecast**.")
    st.subheader("Daily Activity Heatmap")
    if not prophet_forecast.empty:
        calendar_fig = create_calendar_heatmap(prophet_forecast, retailer)
        st.plotly_chart(calendar_fig, use_container_width=True)
    st.subheader("Hourly Activity Heatmap")
    if not prophet_forecast.empty:
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