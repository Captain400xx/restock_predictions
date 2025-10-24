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

