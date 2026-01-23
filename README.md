AI-Powered Sales Forecast Simulator
📊 Project Overview
An intelligent business analytics platform that combines advanced time series forecasting, Monte Carlo simulations, and machine learning to help businesses predict revenue, simulate scenarios, and make data-driven decisions. This project demonstrates end-to-end data science capabilities—from data processing and ML model development to building production-ready applications with intuitive user interfaces.
Live Demo: https://sales-forecast-simulator-ffl9dmdytqya4x7fgj5mky.streamlit.app/

🎯 Project Motivation
This project was inspired by watching data visualization projects that made complex data accessible and engaging. I wanted to create something that went beyond visualization—a tool that could solve real business problems by answering critical questions like:

"What will our sales look like in 6 months?"
"What happens if we increase marketing spend by 30%?"
"How much risk are we taking with this forecast?"

The challenge was to make enterprise-grade forecasting accessible to everyone, from data scientists to small business owners, without requiring any coding knowledge.

✨ Key Features
1. Advanced Time Series Forecasting

Implements Facebook Prophet algorithm for accurate predictions (1-24 months ahead)
Automatically handles seasonality, trends, and holiday effects
Displays confidence intervals for uncertainty quantification
Adapts to any business cycle or industry

![Sales Forecast Chart]
Screenshot: Historical sales with forecasted trend and confidence intervals
2. Interactive What-If Scenario Planning

Real-time scenario modeling with adjustable parameters:

Marketing spend changes (-50% to +100%)
Seasonal campaign boosts (0% to +50%)
Pricing adjustments (-30% to +30%)


Instant visual feedback showing revenue impact
Side-by-side comparison of base forecast vs. scenario forecast

![What-If Scenarios]
Screenshot: What-If sliders and scenario comparison
3. Monte Carlo Risk Analysis

Runs 100-5,000 simulations to quantify uncertainty
Generates percentile-based outcomes (P10, P25, P50, P75, P90)
Visualizes revenue distribution and confidence ranges
Provides worst-case, most-likely, and best-case scenarios
Helps stakeholders understand and communicate risk

![Monte Carlo Simulation]
Screenshot: Monte Carlo simulation results with percentile bands
4. Automated Anomaly Detection

Statistical Z-score methodology to identify unusual patterns
Highlights spikes and drops in sales data
Adjustable sensitivity for different business contexts
Provides actionable alerts for investigation
Helps catch data quality issues or unexpected market changes

![Anomaly Detection]
Screenshot: Anomaly alerts showing unusual sales patterns
5. ML-Powered Feature Importance Analysis

Random Forest model reveals what drives revenue
Works with any dataset structure (not hardcoded)
Analyzes impact of products, regions, segments, and time patterns
Visual ranking of top sales drivers
Generates data-driven recommendations

![Feature Importance]
Screenshot: Feature importance bar chart
6. Universal Data Support

Accepts multiple formats: CSV, Excel, JSON, Parquet, TXT
Intelligent column detection (auto-maps dates and sales)
Handles datasets with or without date columns
Works with minimal or extensive feature sets
Robust error handling for missing/invalid data

7. Professional Reporting

Export forecasts and scenarios to Excel
Generate executive summary reports (TXT format)
Download-ready for stakeholder presentations
Timestamped filenames for version control

![Export Options]
Screenshot: Export buttons and sample report

🛠️ Technical Implementation
Architecture
Data Input → Processing & Validation → ML Models → Interactive Dashboard → Reports
Tech Stack
ComponentTechnologyPurposeFrontend/UIStreamlitInteractive web interfaceTime SeriesFacebook ProphetForecasting with seasonalityMachine LearningScikit-learn (Random Forest)Feature importance analysisData ProcessingPandas, NumPyETL and transformationsVisualizationPlotlyInteractive chartsStatisticsCustom implementationsMonte Carlo, Z-score anomaly detectionExportOpenPyXLExcel report generation
Data Flow

Upload/Selection: User uploads dataset or uses sample data
Validation: Automatic column detection and data quality checks
Filtering: Dynamic filters update all downstream analyses
Forecasting: Prophet model trains on historical data
Scenario Application: User adjustments modify forecast in real-time
Risk Modeling: Monte Carlo simulations run on scenario forecast
Insights: Anomalies detected and feature importance calculated
Export: Reports generated with all findings
