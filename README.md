Version 5.0 (Beta) – Major Overhaul

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
