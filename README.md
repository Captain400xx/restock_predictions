# Pokémon Card Restock Forecasting App

*This Streamlit application uses a combination of time-series and machine learning models to forecast Pokémon TCG restocks from various online retailers. It's designed to identify high-probability restock windows by finding a consensus among four different forecasting models.*

**Features**

Multi-Model Ensemble: Combines predictions from Prophet, LightGBM, XGBoost, and CatBoost for higher confidence.

Prediction Schedule: A color-coded schedule that highlights upcoming restock windows and ranks them by confidence level (High, Medium, Low).

2-Week Accuracy View: A special chart that compares the model's forecast for the past week against actual historical data, providing a visual check on accuracy.

In-Depth Analysis: Includes heatmaps to visualize the most active days of the month and hours of the day for restocks.

Feature Importance: Shows which factors (e.g., time of day, day of week) are most influential in making predictions.

Customizable UI: Includes a simple color picker to customize chart appearances.

**How to Run Locally**

Prerequisites

Python 3.9+

pip for package installation

Clone the Repository
Clone this repository to your local machine.

Set Up Your Data
You must have a CSV file containing your historical restock data.

The file must be named my_restock_data.csv.

It must have at least two columns: DateTime and Retailer.

Place this file in the same root directory as the app.py script.

Install Dependencies
Navigate to the project directory in your terminal and run:

pip install -r requirements.txt

Run the App
Once the dependencies are installed, run the following command:

streamlit run app.py


# Version History:

**Version 5.0 (Beta) – Major Overhaul**

Release Date: October 2025

🔧 Core Model & Forecasting Updates

Improved Prophet Model Training: Fixed weights compatibility and optimized runtime performance.

Outlier Handling: Added automatic capping (99th percentile) for all model forecasts to remove extreme spikes.

Consistent Time Localization: All forecasts now display in the user-selected timezone with 15-minute precision.

Refactored Forecast Function: Unified model output formatting (Prophet, LightGBM, XGBoost, CatBoost).

Dynamic Zoom Feature: Chart automatically focuses on predicted high-activity windows.

Enhanced Big Restock Detection: Introduced get_big_restocks() for standardized event flagging across models.

🎨 UI / Visualization Upgrades

New “How to Use” Guide: Simplified onboarding instructions with clear model explanations.

Refreshed Analysis Section:

Removed redundant daily heatmap view.

Replaced with a single, redesigned Hourly Activity Heatmap inspired by NYC transit-style visuals.

Custom Color Palette: Soft mint-to-white-to-lavender gradient to match the Restock Predictions logo theme.

Cleaner Layouts: Reworked titles, margins, and hover labels for readability on dark mode.

⚙️ Technical Improvements

Fixed multiple layout errors (titlefont and colorbar updates) for full Plotly v5+ compatibility.

Streamlined figure rendering and caching for faster refresh and smoother interactivity.

Added color normalization and legend calibration for consistent chart interpretation.



**v4.0 - Major Accuracy & UI Overhaul This version introduces a significant upgrade to the underlying models for improved accuracy and adds several new UI features for better analysis and usability.**

🧠 Major Model Overhaul Advanced Feature Engineering: The ML models (LightGBM, XGBoost, CatBoost) now use a much richer feature set for higher accuracy, including:

Lag Features: To teach models about momentum (e.g., activity_24_hours_ago).

Cyclical Features: To help models understand the circular nature of time (e.g., that 11 PM is next to 12 AM).

Contextual Features: To provide human context (e.g., is_weekend, is_business_hours), which helps prevent incorrect predictions like weekend restocks for retailers who operate on weekdays only.

Poisson Objective: The models have been retuned with a 'Poisson' objective, which is statistically better for predicting the frequency of spikes and rare events.

✨ New Features & UI Enhancements New Tab: "2-Week View": A powerful new chart that overlaps the past 7 days of actual data with the model's forecast, allowing for an immediate visual check on prediction accuracy.

New "Prediction Schedule" Chart: The main tab now includes a visual bar chart of upcoming predictions, color-coded by confidence level.

New "Data Summary": The sidebar now displays key statistics for the selected retailer, including the date range of available data and the total number of recorded events.

New "Chart Color Picker": A control has been added to the sidebar to allow customization of the main forecast line color.

Credits This application was created and is maintained by Captain400x.

