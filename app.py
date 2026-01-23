import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
import json

# Page config
st.set_page_config(
    page_title="Sales Forecast Simulator",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric label {
        color: #262730 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0e1117 !important;
        font-size: 24px !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #0e1117 !important;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2ca02c;
        padding-top: 20px;
    }
    h3 {
        color: #262730;
    }
    .anomaly-alert {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        background-color: #fff3cd !important;
        border-left: 5px solid #ffc107;
    }
    .anomaly-alert strong {
        color: #856404 !important;
        font-size: 16px;
        display: block;
        margin-bottom: 5px;
    }
    .anomaly-alert br {
        line-height: 1.8;
    }
    .anomaly-alert em {
        color: #856404 !important;
        font-style: italic;
    }
    .anomaly-text {
        color: #856404 !important;
        font-size: 14px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# ============= HELPER FUNCTIONS =============

# Create sample data function (SAFE VERSION)
@st.cache_data
def create_sample_data():
    """Create comprehensive sample dataset"""
    np.random.seed(42)
    
    # Generate 365 days of data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create synthetic sales with trends and seasonality
    base_sales = 1000
    trend = np.linspace(0, 0.3, len(dates))
    seasonality = 0.3 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    weekly_pattern = 0.1 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
    noise = np.random.normal(0, 0.1, len(dates))
    
    sales = base_sales * (1 + trend + seasonality + weekly_pattern + noise)
    
    # Add some spikes for anomalies
    spike_indices = [50, 150, 250, 320]
    for idx in spike_indices:
        if idx < len(sales):
            sales[idx] *= 1.8
    
    # Create DataFrame
    df = pd.DataFrame({
        'Order Date': dates,
        'Sales': sales,
        'Category': np.random.choice(['Furniture', 'Office Supplies', 'Technology', 'Electronics'], len(dates), p=[0.3, 0.3, 0.25, 0.15]),
        'Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], len(dates), p=[0.5, 0.3, 0.2]),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'Profit': sales * np.random.uniform(0.15, 0.35, len(dates)),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], len(dates)),
        'State': np.random.choice(['NY', 'CA', 'IL', 'TX', 'AZ'], len(dates)),
        'Product': np.random.choice(['Chair', 'Desk', 'Computer', 'Phone', 'Tablet'], len(dates))
    })
    
    return df

def detect_date_column(df):
    """Automatically detect date column"""
    date_keywords = ['date', 'time', 'day', 'month', 'year', 'period', 'fecha', 'datum']
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            try:
                pd.to_datetime(df[col].head())
                return col
            except:
                pass
        
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head())
                return col
            except:
                pass
    return None

def detect_sales_column(df):
    """Automatically detect sales/revenue column"""
    sales_keywords = ['sales', 'revenue', 'amount', 'total', 'price', 'value', 'ventas', 'ingreso']
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in sales_keywords):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[0] if len(numeric_cols) > 0 else None

def load_uploaded_file(uploaded_file):
    """Load any file format and return DataFrame"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                except:
                    df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
        
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        
        elif file_extension == 'txt':
            content = uploaded_file.read().decode('utf-8')
            from io import StringIO
            
            # Try different delimiters
            for delimiter in [',', '\t', ';', '|']:
                try:
                    test_df = pd.read_csv(StringIO(content), sep=delimiter, nrows=5)
                    if test_df.shape[1] > 1:
                        df = pd.read_csv(StringIO(content), sep=delimiter)
                        break
                except:
                    continue
            else:
                return None, "Could not parse text file. Please use CSV format."
        else:
            return None, f"Unsupported file format: {file_extension}"
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        return df, None
    
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

# Anomaly Detection Function
def detect_anomalies(df_time_series, sensitivity=2.5):
    """
    Detect anomalies using statistical method (Z-score)
    Returns DataFrame with anomaly flags and scores
    """
    df = df_time_series.copy()
    
    if len(df) < 30:
        # For small datasets, use simple statistics
        mean_val = df['Sales'].mean()
        std_val = df['Sales'].std()
        if std_val == 0:
            std_val = 1
        df['z_score'] = np.abs((df['Sales'] - mean_val) / std_val)
        df['is_anomaly'] = df['z_score'] > sensitivity
        return df
    
    df['rolling_mean'] = df['Sales'].rolling(window=30, center=True, min_periods=1).mean()
    df['rolling_std'] = df['Sales'].rolling(window=30, center=True, min_periods=1).std()
    
    # Handle NaN values
    df['rolling_mean'] = df['rolling_mean'].fillna(df['Sales'].mean())
    df['rolling_std'] = df['rolling_std'].fillna(df['Sales'].std())
    
    # Avoid division by zero
    df['rolling_std'] = df['rolling_std'].replace(0, 1)
    
    df['z_score'] = np.abs((df['Sales'] - df['rolling_mean']) / df['rolling_std'])
    df['z_score'] = df['z_score'].fillna(0)
    df['is_anomaly'] = df['z_score'] > sensitivity
    
    return df

# Simple forecasting function (replaces Prophet)
def simple_forecast(df, periods=180):
    """Simple forecasting using moving average and trend"""
    if len(df) < 30:
        return None
    
    df_forecast = df.copy()
    df_forecast = df_forecast.sort_values('ds')
    
    # Calculate moving average and trend
    ma_window = min(30, len(df_forecast) // 2)
    df_forecast['moving_avg'] = df_forecast['y'].rolling(window=ma_window).mean()
    
    # Simple linear trend
    x = np.arange(len(df_forecast))
    y = df_forecast['y'].values
    trend_coef = np.polyfit(x, y, 1)
    
    # Forecast future values
    last_date = df_forecast['ds'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    
    # Extend trend
    future_x = np.arange(len(df_forecast), len(df_forecast) + periods)
    future_trend = trend_coef[0] * future_x + trend_coef[1]
    
    # Add seasonality (simple sine wave)
    seasonal_component = 0.1 * np.sin(2 * np.pi * future_x / 365) * np.mean(y)
    
    forecast_values = future_trend + seasonal_component
    uncertainty = np.std(y) * 0.3  # 30% uncertainty
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values,
        'yhat_lower': forecast_values * 0.85,
        'yhat_upper': forecast_values * 1.15,
        'trend': future_trend
    })
    
    return forecast_df

def calculate_feature_importance(df):
    """
    Simple feature importance calculation without sklearn
    """
    importance_dict = {}
    
    # Time-based features
    if 'Order Date' in df.columns:
        df['Month'] = df['Order Date'].dt.month
        df['DayOfWeek'] = df['Order Date'].dt.dayofweek
        df['Quarter'] = df['Order Date'].dt.quarter
        
        # Calculate correlation with sales
        for feature in ['Month', 'DayOfWeek', 'Quarter']:
            if feature in df.columns:
                corr = abs(df[feature].corr(df['Sales']))
                if not pd.isna(corr):
                    importance_dict[feature] = corr
    
    # Categorical features
    categorical_cols = ['Category', 'Segment', 'Region', 'City', 'State', 'Product']
    for col in categorical_cols:
        if col in df.columns:
            try:
                # Calculate mean sales by category as importance proxy
                group_means = df.groupby(col)['Sales'].mean()
                if len(group_means) > 1:
                    # Use coefficient of variation as importance measure
                    importance = group_means.std() / group_means.mean()
                    if not pd.isna(importance):
                        importance_dict[col] = importance
            except:
                pass
    
    # Normalize importance scores
    if importance_dict:
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
    
    return importance_dict

def create_excel_report(df, forecast_data, scenario_data, metrics):
    """Create Excel report with multiple sheets"""
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Historical Data
            hist_cols = ['Order Date', 'Sales']
            optional_cols = ['Category', 'Segment', 'Region', 'Profit', 'City', 'State', 'Product']
            for col in optional_cols:
                if col in df.columns:
                    hist_cols.append(col)
            
            df[hist_cols].head(1000).to_excel(writer, sheet_name='Historical Data', index=False)
            
            # Sheet 3: Forecast
            if forecast_data is not None:
                forecast_data.to_excel(writer, sheet_name='Forecast', index=False)
            
            # Sheet 4: Scenario Analysis
            if scenario_data is not None:
                scenario_data.to_excel(writer, sheet_name='Scenario Analysis', index=False)
        
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error creating Excel report: {str(e)}")
        return None

def create_pdf_summary(metrics, insights):
    """Create a simple text-based summary report"""
    report = f"""
SALES FORECAST REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

==================================================
EXECUTIVE SUMMARY
==================================================

Total Revenue: ${metrics.get('Total Revenue', 0):,.2f}
Total Profit: ${metrics.get('Total Profit', 0):,.2f}
Forecast Period: {metrics.get('Forecast Period', 'N/A')} months
Base Forecast: ${metrics.get('Base Forecast', 0):,.2f}
Scenario Forecast: ${metrics.get('Scenario Forecast', 0):,.2f}
Revenue Impact: ${metrics.get('Revenue Impact', 0):,.2f}

==================================================
KEY INSIGHTS
==================================================

{insights}

==================================================
RISK ANALYSIS
==================================================

Risk Level: {metrics.get('Risk Level', 'N/A')}
Downside Risk: {metrics.get('Downside Risk', 0):.1f}%
Upside Potential: {metrics.get('Upside Potential', 0):.1f}%
Confidence Range: ${metrics.get('P10', 0):,.0f} - ${metrics.get('P90', 0):,.0f}

==================================================
End of Report
==================================================
    """
    return report

# Monte Carlo simulation
def monte_carlo_simulation(base_values, num_simulations=1000, uncertainty=0.15):
    """Simple Monte Carlo simulation"""
    np.random.seed(42)
    simulations = []
    
    for _ in range(num_simulations):
        # Add random noise to base values
        noise = np.random.normal(1.0, uncertainty, len(base_values))
        sim_result = base_values * noise
        simulations.append(sim_result)
    
    simulations = np.array(simulations)
    
    # Calculate percentiles
    percentiles = {
        'P10': np.percentile(simulations, 10, axis=0),
        'P25': np.percentile(simulations, 25, axis=0),
        'P50': np.percentile(simulations, 50, axis=0),
        'P75': np.percentile(simulations, 75, axis=0),
        'P90': np.percentile(simulations, 90, axis=0)
    }
    
    return simulations, percentiles

# ============= DATA LOADING =============
st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use Sample Dataset", "Upload Your Own File"]
)

if data_source == "Upload Your Own File":
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your sales data",
        type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        help="Supported: CSV, Excel, JSON, TXT"
    )
    
    if uploaded_file is not None:
        with st.spinner('Loading and analyzing your file...'):
            df, error = load_uploaded_file(uploaded_file)
            
            if error:
                st.error(error)
                st.info("Please use the sample dataset or upload a different file format.")
                use_sample = st.button("Use Sample Dataset Instead")
                if use_sample:
                    df = create_sample_data()
                else:
                    st.stop()
            else:
                st.sidebar.success(f"File loaded successfully: {len(df):,} rows, {df.shape[1]} columns")
                
                # Show column preview
                st.sidebar.subheader("Column Preview")
                st.sidebar.write(f"Columns: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}")
                
                # Automatically detect columns
                date_col = detect_date_column(df)
                sales_col = detect_sales_column(df)
                
                # Handle date column
                if date_col:
                    try:
                        df['Order Date'] = pd.to_datetime(df[date_col], errors='coerce')
                        valid_dates = df['Order Date'].notna().sum()
                        if valid_dates > 0:
                            st.sidebar.info(f"Using '{date_col}' as date column ({valid_dates:,} valid dates)")
                        else:
                            st.sidebar.warning(f"'{date_col}' doesn't contain valid dates. Using generated dates.")
                            df['Order Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
                    except Exception as e:
                        st.sidebar.warning(f"Could not parse '{date_col}' as dates: {str(e)}. Using generated dates.")
                        df['Order Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
                else:
                    st.sidebar.info("No date column detected. Using generated timeline.")
                    df['Order Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
                
                # Handle sales column
                if sales_col:
                    df['Sales'] = pd.to_numeric(df[sales_col], errors='coerce')
                    valid_sales = df['Sales'].notna().sum()
                    if valid_sales > 0:
                        st.sidebar.info(f"Using '{sales_col}' as sales column ({valid_sales:,} valid values)")
                        df = df[df['Sales'].notna()].copy()
                    else:
                        st.sidebar.error("No valid sales data found in the selected column.")
                        st.stop()
                else:
                    st.sidebar.error("No numeric column found for sales data.")
                    st.stop()
                
                # Add required columns if missing
                required_cols = ['Category', 'Segment', 'Region', 'Profit']
                for col in required_cols:
                    if col not in df.columns:
                        if col == 'Profit':
                            df[col] = df['Sales'] * 0.2  # Default 20% profit margin
                        else:
                            df[col] = 'All'
                
                # Try to map existing columns
                if 'Category' not in df.columns or df['Category'].iloc[0] == 'All':
                    for col in df.columns:
                        if col.lower() in ['category', 'type', 'product_type', 'department']:
                            df['Category'] = df[col]
                            break
                
                if 'Segment' not in df.columns or df['Segment'].iloc[0] == 'All':
                    for col in df.columns:
                        if col.lower() in ['segment', 'customer_type', 'client_type']:
                            df['Segment'] = df[col]
                            break
                
                if 'Region' not in df.columns or df['Region'].iloc[0] == 'All':
                    for col in df.columns:
                        if col.lower() in ['region', 'area', 'territory', 'location']:
                            df['Region'] = df[col]
                            break
    
    else:
        st.info("Please upload a file to begin")
        # Use sample data as fallback
        df = create_sample_data()

else:
    # Load sample dataset
    df = create_sample_data()

# Header
st.title("Sales Forecast Simulator")
st.markdown("### Interactive Sales Forecasting with What-If Scenarios & Risk Analysis")
st.markdown("---")

# ============= SIDEBAR FILTERS & CONTROLS =============
st.sidebar.header("Control Panel")

st.sidebar.subheader("Data Filters")
category_filter = st.sidebar.multiselect(
    "Product Category",
    options=sorted(df['Category'].unique()),
    default=df['Category'].unique()[:min(3, len(df['Category'].unique()))]
)

segment_filter = st.sidebar.multiselect(
    "Customer Segment",
    options=sorted(df['Segment'].unique()),
    default=df['Segment'].unique()
)

region_filter = st.sidebar.multiselect(
    "Region",
    options=sorted(df['Region'].unique()),
    default=df['Region'].unique()
)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Settings")
forecast_months = st.sidebar.slider("Forecast Period (months)", 1, 24, 6)

st.sidebar.markdown("---")
st.sidebar.subheader("What-If Scenarios")

marketing_boost = st.sidebar.slider(
    "Marketing Spend (%)",
    min_value=-50,
    max_value=100,
    value=0,
    step=5
)

seasonality_boost = st.sidebar.slider(
    "Seasonal Boost (%)",
    min_value=0,
    max_value=50,
    value=0,
    step=5
)

pricing_change = st.sidebar.slider(
    "Price Change (%)",
    min_value=-30,
    max_value=30,
    value=0,
    step=5
)

st.sidebar.markdown("---")
st.sidebar.subheader("Monte Carlo Settings")

num_simulations = st.sidebar.select_slider(
    "Number of Simulations",
    options=[100, 500, 1000, 2000, 5000],
    value=1000
)

uncertainty_level = st.sidebar.slider(
    "Uncertainty Level (%)",
    min_value=5,
    max_value=30,
    value=15,
    step=5
)

st.sidebar.markdown("---")
st.sidebar.subheader("Anomaly Detection")
anomaly_sensitivity = st.sidebar.slider(
    "Sensitivity (higher = fewer alerts)",
    min_value=1.5,
    max_value=4.0,
    value=2.5,
    step=0.5
)

# ============= APPLY FILTERS =============
filtered_df = df.copy()
if category_filter:
    filtered_df = filtered_df[filtered_df['Category'].isin(category_filter)]
if segment_filter:
    filtered_df = filtered_df[filtered_df['Segment'].isin(segment_filter)]
if region_filter:
    filtered_df = filtered_df[filtered_df['Region'].isin(region_filter)]

# ============= KEY METRICS SECTION =============
st.subheader("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

total_revenue = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum() if 'Profit' in filtered_df.columns else filtered_df['Sales'].sum() * 0.2
total_orders = len(filtered_df)
avg_order_value = filtered_df['Sales'].mean()
profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0

col1.metric("Total Revenue", f"${total_revenue:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Total Orders", f"{total_orders:,}")
col4.metric("Avg Order Value", f"${avg_order_value:.2f}")
col5.metric("Profit Margin", f"{profit_margin:.1f}%")

st.markdown("---")

# ============= FEATURE IMPORTANCE ANALYSIS =============
st.subheader("Feature Importance Analysis")
st.markdown("Understanding what drives your sales")

with st.spinner('Analyzing feature importance...'):
    importance_dict = calculate_feature_importance(filtered_df)
    
    if importance_dict:
        # Sort by importance
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        features_sorted = [item[0] for item in sorted_items]
        importances_sorted = [item[1] for item in sorted_items]
        
        # Plot feature importance
        fig_importance = go.Figure()
        
        # Color top 3 differently
        colors = ['#2ca02c' if i < 3 else '#1f77b4' for i in range(len(features_sorted))]
        
        fig_importance.add_trace(go.Bar(
            x=importances_sorted,
            y=features_sorted,
            orientation='h',
            marker_color=colors,
            text=[f'{imp*100:.1f}%' for imp in importances_sorted],
            textposition='auto'
        ))
        
        fig_importance.update_layout(
            title='Feature Importance: What Drives Your Sales?',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Drivers:**")
            top_n = min(3, len(features_sorted))
            for i in range(top_n):
                st.markdown(f"{i+1}. **{features_sorted[i]}**: {importances_sorted[i]*100:.1f}% impact on sales")
        
        with col2:
            st.markdown("**Recommendations:**")
            if features_sorted:
                top_feature = features_sorted[0]
                
                if top_feature == 'Category':
                    st.markdown("- Focus marketing on high-performing product categories")
                    st.markdown("- Analyze category-specific trends for inventory planning")
                elif top_feature == 'Segment':
                    st.markdown("- Target customer segments more strategically")
                    st.markdown("- Develop segment-specific marketing campaigns")
                elif top_feature == 'Region':
                    st.markdown("- Optimize regional distribution and marketing")
                    st.markdown("- Consider regional preferences in product mix")
                elif top_feature == 'Month':
                    st.markdown("- Strong seasonal patterns detected")
                    st.markdown("- Plan inventory and staffing around peak months")
                elif top_feature == 'Quarter':
                    st.markdown("- Quarterly trends are significant")
                    st.markdown("- Align business planning with quarterly cycles")
                else:
                    st.markdown(f"- **{top_feature}** is your strongest sales driver")
                    st.markdown("- Focus resources on optimizing this factor")
                
                if top_n > 1:
                    st.markdown(f"- Top {top_n} factors account for {sum(importances_sorted[:top_n])*100:.1f}% of sales variance")
    else:
        st.info("Not enough feature variation for importance analysis. Try adjusting your filters or ensure your dataset has categorical columns.")

st.markdown("---")

# ============= HISTORICAL SALES WITH ANOMALY DETECTION =============
st.subheader("Historical Sales Analysis with Anomaly Detection")

if 'Order Date' in filtered_df.columns:
    df_daily = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
    
    if len(df_daily) > 0:
        df_anomaly = detect_anomalies(df_daily, sensitivity=anomaly_sensitivity)
        
        # Plot with anomalies
        fig_historical = go.Figure()

        # Normal sales
        fig_historical.add_trace(go.Scatter(
            x=df_anomaly['Order Date'],
            y=df_anomaly['Sales'],
            mode='lines',
            name='Daily Sales',
            line=dict(color='#1f77b4', width=2)
        ))

        # Anomalies
        anomalies = df_anomaly[df_anomaly['is_anomaly']]
        if len(anomalies) > 0:
            fig_historical.add_trace(go.Scatter(
                x=anomalies['Order Date'],
                y=anomalies['Sales'],
                mode='markers',
                name='Anomalies Detected',
                marker=dict(color='red', size=10, symbol='circle')
            ))

        fig_historical.update_layout(
            title='Daily Sales Trend with Anomaly Detection',
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig_historical, use_container_width=True)

        # Anomaly alerts
        if len(anomalies) > 0:
            st.markdown("### Anomaly Alerts")
            
            # Show top 5 most recent anomalies
            recent_anomalies = anomalies.sort_values('Order Date', ascending=False).head(5)
            
            for _, row in recent_anomalies.iterrows():
                date = row['Order Date'].strftime('%Y-%m-%d')
                sales = row['Sales']
                z_score = row['z_score']
                
                if sales > row['rolling_mean'] if 'rolling_mean' in row else sales > df_anomaly['Sales'].mean():
                    alert_type = "Unusual Spike"
                else:
                    alert_type = "Unusual Drop"
                
                st.markdown(f"""
                <div class="anomaly-alert">
                    <strong>{alert_type} detected on {date}</strong>
                    <div class="anomaly-text">
                        Sales: ${sales:,.2f} (Z-score: {z_score:.2f})<br>
                        <em>Investigate: Check for promotions, holidays, or data errors</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant anomalies detected in the current data filter")
    else:
        st.warning("No data available after filtering for anomaly detection.")
else:
    st.warning("Date column not available for time series analysis.")

st.markdown("---")

# ============= FORECASTING ENGINE =============
st.subheader("Sales Forecast with What-If Scenarios")

if 'Order Date' in filtered_df.columns and len(filtered_df) > 30:
    # Prepare data for forecasting
    forecast_df = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
    forecast_df.columns = ['ds', 'y']
    forecast_df = forecast_df.sort_values('ds')
    
    with st.spinner('Generating forecast...'):
        forecast_result = simple_forecast(forecast_df, periods=forecast_months * 30)
        
        if forecast_result is not None:
            forecast_future = forecast_result.copy()
            
            # Apply scenarios
            marketing_impact = marketing_boost * 0.5
            seasonality_impact = seasonality_boost
            pricing_impact = pricing_change * -0.7
            total_impact = (100 + marketing_impact + seasonality_impact + pricing_impact) / 100
            
            forecast_future['yhat_scenario'] = forecast_future['yhat'] * total_impact
            forecast_future['yhat_upper_scenario'] = forecast_future['yhat_upper'] * total_impact
            forecast_future['yhat_lower_scenario'] = forecast_future['yhat_lower'] * total_impact
            
            # Plot forecast
            fig_forecast = go.Figure()
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['y'],
                mode='lines',
                name='Historical',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat'],
                mode='lines',
                name='Base Forecast',
                line=dict(color='gray', width=1, dash='dot'),
                opacity=0.5
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat_scenario'],
                mode='lines',
                name='Scenario Forecast',
                line=dict(color='#2ca02c', width=3)
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat_upper_scenario'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat_lower_scenario'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(44, 160, 68, 0.2)',
                line=dict(width=0),
                name='Confidence Interval'
            ))
            
            fig_forecast.update_layout(
                title=f'Sales Forecast - Next {forecast_months} Months (Scenario Applied)',
                xaxis_title='Date',
                yaxis_title='Sales ($)',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Scenario metrics
            col1, col2, col3, col4 = st.columns(4)
            
            base_total = forecast_future['yhat'].sum()
            scenario_total = forecast_future['yhat_scenario'].sum()
            revenue_diff = scenario_total - base_total
            percent_change = ((scenario_total - base_total) / base_total * 100) if base_total > 0 else 0
            
            col1.metric("Base Forecast", f"${base_total:,.0f}")
            col2.metric("Scenario Forecast", f"${scenario_total:,.0f}", f"{percent_change:+.1f}%")
            col3.metric("Revenue Impact", f"${revenue_diff:+,.0f}")
            col4.metric("Total Adjustment", f"{(total_impact - 1) * 100:+.1f}%")
            
            # ============= MONTE CARLO SIMULATION =============
            st.markdown("---")
            st.subheader("Monte Carlo Risk Analysis")
            
            with st.spinner(f'Running {num_simulations:,} Monte Carlo simulations...'):
                simulations, percentiles = monte_carlo_simulation(
                    forecast_future['yhat_scenario'].values,
                    num_simulations,
                    uncertainty_level/100
                )
            
            p10 = percentiles['P10']
            p25 = percentiles['P25']
            p50 = percentiles['P50']
            p75 = percentiles['P75']
            p90 = percentiles['P90']
            
            fig_mc = go.Figure()
            
            fig_mc.add_trace(go.Scatter(
                x=forecast_future['ds'], y=p90,
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            
            fig_mc.add_trace(go.Scatter(
                x=forecast_future['ds'], y=p10,
                mode='lines', fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(width=0), name='10th-90th Percentile'
            ))
            
            fig_mc.add_trace(go.Scatter(
                x=forecast_future['ds'], y=p75,
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            
            fig_mc.add_trace(go.Scatter(
                x=forecast_future['ds'], y=p25,
                mode='lines', fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(width=0), name='25th-75th Percentile'
            ))
            
            fig_mc.add_trace(go.Scatter(
                x=forecast_future['ds'], y=p50,
                mode='lines', name='Median',
                line=dict(color='#2ca02c', width=3)
            ))
            
            fig_mc.update_layout(
                title=f'Monte Carlo Simulation ({num_simulations:,} runs)',
                xaxis_title='Date',
                yaxis_title='Sales ($)',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_mc, use_container_width=True)
            
            st.subheader("Risk Analysis")
            
            total_p10 = p10.sum()
            total_p25 = p25.sum()
            total_p50 = p50.sum()
            total_p75 = p75.sum()
            total_p90 = p90.sum()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("Worst Case (P10)", f"${total_p10:,.0f}")
            col2.metric("Conservative (P25)", f"${total_p25:,.0f}")
            col3.metric("Most Likely (P50)", f"${total_p50:,.0f}")
            col4.metric("Optimistic (P75)", f"${total_p75:,.0f}")
            col5.metric("Best Case (P90)", f"${total_p90:,.0f}")
            
            sim_totals = simulations.sum(axis=1)
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=sim_totals, nbinsx=50,
                marker_color='#2ca02c', opacity=0.7
            ))
            
            # Add percentile lines
            for percentile, value, color in [(10, total_p10, 'red'), (50, total_p50, 'green'), (90, total_p90, 'orange')]:
                fig_hist.add_vline(
                    x=value, line_dash="dash", line_color=color,
                    annotation_text=f"P{percentile}: ${value:,.0f}"
                )

            fig_hist.update_layout(
                title='Revenue Distribution',
                xaxis_title='Total Revenue ($)',
                yaxis_title='Frequency',
                showlegend=False,
                height=350
            )

            st.plotly_chart(fig_hist, use_container_width=True)
            
            # ============= EXPORT & REPORTING =============
            st.markdown("---")
            st.subheader("Export & Reporting")
            
            downside_risk = (scenario_total - total_p10) / scenario_total * 100 if scenario_total > 0 else 0
            upside_potential = (total_p90 - scenario_total) / scenario_total * 100 if scenario_total > 0 else 0
            risk_level = "LOW" if downside_risk < 15 else "MODERATE" if downside_risk < 25 else "HIGH"
            
            # Prepare data for export
            export_metrics = {
                'Total Revenue': total_revenue,
                'Total Profit': total_profit,
                'Forecast Period': f"{forecast_months} months",
                'Base Forecast': base_total,
                'Scenario Forecast': scenario_total,
                'Revenue Impact': revenue_diff,
                'Risk Level': risk_level,
                'Downside Risk': downside_risk,
                'Upside Potential': upside_potential,
                'P10': total_p10,
                'P50': total_p50,
                'P90': total_p90
            }
            
            insights_text = f"""Top Sales Driver: {features_sorted[0] if 'features_sorted' in locals() and features_sorted else 'N/A'} ({importances_sorted[0]*100:.1f}% impact if available)
Anomalies Detected: {len(anomalies) if 'anomalies' in locals() else 0}
Marketing Adjustment: {marketing_boost:+}%
Seasonal Boost: {seasonality_boost:+}%
Price Adjustment: {pricing_change:+}%
Total Forecast Impact: {(total_impact - 1) * 100:+.1f}%
"""
            forecast_export_df = forecast_future[['ds', 'yhat', 'yhat_scenario', 'yhat_lower_scenario', 'yhat_upper_scenario']].copy()
            forecast_export_df.columns = ['Date', 'Base Forecast', 'Scenario Forecast', 'Lower Bound', 'Upper Bound']

            scenario_export_df = pd.DataFrame({
                'Percentile': ['P10', 'P25', 'P50', 'P75', 'P90'],
                'Total Revenue': [total_p10, total_p25, total_p50, total_p75, total_p90]
            })

            col1, col2 = st.columns(2)

            with col1:
                # Excel Export
                excel_data = create_excel_report(
                    filtered_df,
                    forecast_export_df,
                    scenario_export_df,
                    export_metrics
                )
                
                if excel_data:
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name=f"sales_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            with col2:
                # PDF/Text Export
                pdf_content = create_pdf_summary(export_metrics, insights_text)
                
                st.download_button(
                    label="Download Summary Report (TXT)",
                    data=pdf_content,
                    file_name=f"sales_forecast_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            # ============= KEY INSIGHTS =============
            st.markdown("---")
            st.subheader("Key Insights & Recommendations")

            insight_col1, insight_col2 = st.columns(2)

            with insight_col1:
                st.markdown(f"""
                **Risk Assessment:**
                - Risk Level: **{risk_level}**
                - Downside Risk: **{downside_risk:.1f}%**
                - Upside Potential: **{upside_potential:.1f}%**
                - 80% Confidence Range: **${total_p10:,.0f} - ${total_p90:,.0f}**
                """)

            with insight_col2:
                st.markdown("**Scenario Impact:**")
                if marketing_boost != 0:
                    st.markdown(f"- Marketing: {marketing_boost:+}% → Sales impact: {marketing_impact:+.1f}%")
                if seasonality_boost != 0:
                    st.markdown(f"- Seasonality: +{seasonality_boost}% → Sales boost: +{seasonality_impact:.1f}%")
                if pricing_change != 0:
                    st.markdown(f"- Pricing: {pricing_change:+}% → Volume impact: {pricing_impact:+.1f}%")
                if marketing_boost == 0 and seasonality_boost == 0 and pricing_change == 0:
                    st.markdown("- No adjustments applied (base scenario)")
                
        else:
            st.warning("Forecasting requires more data. Please upload a larger dataset or use the sample data.")
else:
    st.warning("Forecasting requires date column and sufficient historical data (minimum 30 records).")

st.markdown("---")
st.markdown("Built with Streamlit | Data updates in real-time as you adjust filters")

# ============= DATA PREVIEW =============
with st.expander("View Raw Data Preview"):
    st.dataframe(filtered_df.head(50))
    st.caption(f"Showing 50 of {len(filtered_df):,} total records")