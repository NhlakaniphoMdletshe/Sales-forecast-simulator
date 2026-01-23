# AI-Powered Sales Forecast Simulator 📊

An intelligent business analytics platform that combines time series forecasting, Monte Carlo simulations, and machine learning to help businesses predict revenue, simulate scenarios, and make data-driven decisions.

This project demonstrates end-to-end data science skills, from data processing and ML model development to deploying a production-ready interactive application.

👉 **Live Demo:** Try it here  

---

## 🎯 Project Motivation

This project was inspired by data visualization tools that make complex data easier to understand. I wanted to go beyond charts and build something that helps answer real business questions like:

- What will our sales look like in 6 months?
- What happens if we increase marketing spend by 30%?
- How much risk is built into this forecast?

The goal was to make enterprise-level forecasting accessible to everyone, including non-technical users, without requiring any coding knowledge.

---

## ✨ Key Features

### 1. Advanced Time Series Forecasting
- Uses Facebook Prophet for forecasts (1–24 months)
- Automatically captures trends, seasonality, and holidays
- Displays confidence intervals for uncertainty
- Adapts to different industries and business cycles

![Sales Forecast Chart]
_Historical sales with forecasted trend and confidence intervals_

---

### 2. Interactive What-If Scenario Planning
Adjust scenarios in real time:
- Marketing spend (-50% to +100%)
- Seasonal campaign boosts (0% to +50%)
- Pricing adjustments (-30% to +30%)

- Instant visual feedback
- Side-by-side comparison of base vs scenario forecasts

![What-If Scenarios]
_What-If sliders and scenario comparison_

---

### 3. Monte Carlo Risk Analysis
- Runs 100–5,000 simulations
- Percentile outcomes (P10, P25, P50, P75, P90)
- Revenue distribution visualization
- Worst-case, most-likely, and best-case scenarios

![Monte Carlo Simulation]
_Monte Carlo simulation results_

---

### 4. Automated Anomaly Detection
- Z-score based statistical detection
- Flags unusual spikes and drops
- Adjustable sensitivity
- Helps catch data issues or unexpected market behavior

![Anomaly Detection]
_Anomaly alerts in sales data_

---

### 5. ML-Powered Feature Importance
- Random Forest model identifies revenue drivers
- Works with flexible dataset structures
- Supports numeric and categorical features
- Clear visual ranking of drivers

![Feature Importance]
_Feature importance bar chart_

---

### 6. Universal Data Support
- CSV, Excel, JSON, Parquet, TXT
- Automatic column detection
- Works with or without date columns
- Robust handling of missing or invalid data

---

### 7. Professional Reporting
- Export forecasts and scenarios to Excel
- Generate executive summary reports (TXT)
- Timestamped filenames for version control

![Export Options]
_Export buttons and sample reports_

---

## 🛠️ Technical Implementation

### Architecture
