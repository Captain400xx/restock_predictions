# 🎴 Pokémon Card Restock Forecasting App

*This Streamlit application combines time-series and machine learning models to forecast Pokémon TCG restocks across multiple online retailers. It identifies high-probability restock windows by finding consensus among four predictive models.*

---

## 🚀 Features

- **Multi-Model Ensemble:** Combines predictions from **Prophet**, **LightGBM**, **XGBoost**, and **CatBoost** for higher confidence.  
- **Prediction Schedule:** A color-coded schedule highlighting upcoming restock windows ranked by confidence (High / Medium / Low).  
- **2-Week Accuracy View:** Compares the past week’s actual data against model forecasts for quick visual accuracy checks.  
- **In-Depth Analysis:** Interactive heatmaps reveal the most active days and hours for restocks.  
- **Feature Importance:** Displays which factors (time of day, weekday, etc.) most influence predictions.  
- **Customizable UI:** Includes an easy color picker for customizing chart themes.  

---

## 🧩 How to Run Locally

### **Prerequisites**
- Python **3.9+**
- **pip** for package installation

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/restock_forecast_app.git
```

## Set Up Your Data

- Place your historical restock data in the same directory as the app:
- File name: my_restock_data.csv
- Required columns: DateTime, Retailer

## Install Dependencies
```
pip install -r requirements.txt
```

## Run the App
```
streamlit run app.py
```

## 🧾 Version History

### Version 5.0 (Beta) – Major Overhaul

*Release Date: October 2025*

**🔧 Core Model & Forecasting Updates**

- Improved Prophet model training — fixed weights compatibility and optimized runtime.
- Automatic outlier capping (99th percentile) for smoother forecasts.
- Consistent time localization (user-selected timezone, 15-min precision).
- Unified model outputs (Prophet, LightGBM, XGBoost, CatBoost).
- Dynamic zoom to focus on high-activity windows.
- New get_big_restocks() function for consistent event flagging.

**🎨 UI / Visualization Upgrades**

- Added “How to Use” guide for onboarding.
- Replaced redundant daily heatmap with redesigned Hourly Activity Heatmap (NYC-transit-inspired).
- Introduced mint-to-white-to-lavender color palette matching the Restock Predictions logo.
- Reworked layouts, titles, and hover labels for better dark-mode readability.

**⚙️ Technical Improvements**

- Fixed layout and Plotly v5+ compatibility issues.
- Streamlined caching and rendering for faster performance.
- Added color normalization and legend calibration for consistent chart interpretation.

### Version 4.0 – Accuracy & UI Overhaul

**🧠 Model Enhancements**

- Advanced Feature Engineering
- Lag features (e.g., activity_24_hours_ago)
- Cyclical time features (e.g., hour-of-day)
- Contextual features (is_weekend, is_business_hours)
- Poisson Objective: Better for forecasting rare event frequencies.

**✨ New Features**

- 2-Week View: Compare model forecasts vs. actual results.
- Prediction Schedule: Visual bar chart ranked by confidence level.
- Data Summary Sidebar: Key dataset stats (date range, total events).
- Color Picker: Customize forecast line color.

## 👑 Credits

Created and maintained by **Captain400x**

*📫 For inquiries or collaboration, visit github.com/Captain400xx*
