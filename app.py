import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import json
from io import BytesIO
from datetime import datetime
import base64

import plotly
import plotly.express as px
print("Plotly version:", plotly.__version__)


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

# Universal File Loader Functions
def detect_date_column(df):
    """Automatically detect date column"""
    date_keywords = ['date', 'time', 'day', 'month', 'year', 'period', 'fecha', 'datum']
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            try:
                pd.to_datetime(df[col])
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
                    df = pd.read_csv(uploaded_file, encoding='iso-8859-1', delimiter=';')
        
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        
        elif file_extension == 'txt':
            content = uploaded_file.read().decode('utf-8')
            from io import StringIO
            for delimiter in [',', '\t', ';', '|']:
                try:
                    df = pd.read_csv(StringIO(content), delimiter=delimiter)
                    if df.shape[1] > 1:
                        break
                except:
                    continue
        
        else:
            return None, f"Unsupported file format: {file_extension}"
        
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
    df['rolling_mean'] = df['Sales'].rolling(window=30, center=True).mean()
    df['rolling_std'] = df['Sales'].rolling(window=30, center=True).std()
    
    df['z_score'] = np.abs((df['Sales'] - df['rolling_mean']) / df['rolling_std'])
    df['is_anomaly'] = df['z_score'] > sensitivity
    
    return df

# Feature Importance Analysis Function (UPDATED - NOT USED ANYMORE, but keeping for compatibility)
def calculate_feature_importance(df):
    """
    Calculate which features most impact sales - DYNAMIC VERSION
    Works with any dataset structure
    """
    df_model = df.copy()
    
    # Find all categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if df[col].nunique() < 50 and df[col].nunique() > 1]
    
    # Encode categorical variables
    encoded_features = []
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[f'{col}_encoded'] = le.fit_transform(df_model[col].astype(str))
            encoded_features.append(f'{col}_encoded')
    
    # Add time features if Order Date exists
    if 'Order Date' in df_model.columns:
        df_model['Month'] = df_model['Order Date'].dt.month
        df_model['DayOfWeek'] = df_model['Order Date'].dt.dayofweek
        df_model['Quarter'] = df_model['Order Date'].dt.quarter
        time_features = ['Month', 'DayOfWeek', 'Quarter']
    else:
        time_features = []
    
    # Combine features
    feature_cols = encoded_features + time_features
    
    if len(feature_cols) < 2:
        return {}, df_model
    
    X = df_model[feature_cols]
    y = df_model['Sales']
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    
    # Get feature importance
    importance_dict = {}
    for i, col in enumerate(feature_cols):
        if col.endswith('_encoded'):
            original_name = col.replace('_encoded', '')
        else:
            original_name = col
        importance_dict[original_name] = rf.feature_importances_[i]
    
    return importance_dict, df_model

# Export Functions
def create_excel_report(df, forecast_data, scenario_data, metrics):
    """Create Excel report with multiple sheets"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Summary
        summary_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Historical Data
        df[['Order Date', 'Sales', 'Category', 'Segment', 'Region']].to_excel(
            writer, sheet_name='Historical Data', index=False
        )
        
        # Sheet 3: Forecast
        forecast_data.to_excel(writer, sheet_name='Forecast', index=False)
        
        # Sheet 4: Scenario Analysis
        scenario_data.to_excel(writer, sheet_name='Scenario Analysis', index=False)
    
    output.seek(0)
    return output

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

# ============= DATA LOADING =============
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
        type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'txt'],
        help="Supported: CSV, Excel, JSON, Parquet, TXT"
    )
    
    if uploaded_file is not None:
        with st.spinner('Loading and analyzing your file...'):
            df, error = load_uploaded_file(uploaded_file)
            
            if error:
                st.error(error)
                st.stop()
            
            st.sidebar.success(f"File loaded: {len(df):,} rows")
            
            # Show column preview
            st.sidebar.subheader("Column Mapping")
            st.sidebar.markdown("Map your data columns to required fields:")
            
            # Detect potential date columns (but don't fail if none found)
            date_col = detect_date_column(df)
            sales_col = detect_sales_column(df)
            
            # Check if dataset has any date-like columns
            has_date_columns = False
            date_candidates = []
            
            for col in df.columns:
                try:
                    # Try to parse first few values
                    test_series = pd.to_datetime(df[col].head(10), errors='coerce')
                    if test_series.notna().sum() > 5:  # At least half are valid dates
                        date_candidates.append(col)
                        has_date_columns = True
                except:
                    pass
            
            # Option 1: Dataset HAS date columns
            if has_date_columns or date_col:
                st.sidebar.info("Date column detected - Time series forecasting available")
                
                if date_col and date_col in date_candidates:
                    default_date_idx = date_candidates.index(date_col)
                else:
                    default_date_idx = 0
                
                date_col = st.sidebar.selectbox(
                    "Date Column",
                    options=date_candidates if date_candidates else df.columns,
                    index=default_date_idx if date_candidates else 0
                )
                
                # Try to parse the date column
                try:
                    df['Order Date'] = pd.to_datetime(df[date_col], errors='coerce')
                    
                    # Check how many valid dates we got
                    valid_dates = df['Order Date'].notna().sum()
                    total_rows = len(df)
                    
                    if valid_dates < total_rows * 0.5:  # Less than 50% valid dates
                        st.sidebar.warning(f"Warning: Only {valid_dates}/{total_rows} rows have valid dates. Consider choosing a different column.")
                    
                    # Remove rows with invalid dates
                    df = df[df['Order Date'].notna()].copy()
                    
                except Exception as e:
                    st.sidebar.error(f"Error parsing dates: {str(e)}")
                    st.sidebar.info("Try selecting a different date column or use 'No Date Column' option")
                    st.stop()
            
            # Option 2: Dataset has NO date columns
            else:
                st.sidebar.warning("No date column detected")
                
                date_option = st.sidebar.radio(
                    "How to handle dates?",
                    [
                        "Generate dates (auto-create timeline)",
                        "Use row index as time sequence",
                        "Skip time series analysis (aggregated view only)"
                    ]
                )
                
                if date_option == "Generate dates (auto-create timeline)":
                    start_date = st.sidebar.date_input(
                        "Start Date",
                        value=pd.to_datetime("2023-01-01")
                    )
                    
                    frequency = st.sidebar.selectbox(
                        "Frequency",
                        ["Daily", "Weekly", "Monthly"]
                    )
                    
                    # Generate date range
                    if frequency == "Daily":
                        date_range = pd.date_range(start=start_date, periods=len(df), freq='D')
                    elif frequency == "Weekly":
                        date_range = pd.date_range(start=start_date, periods=len(df), freq='W')
                    else:  # Monthly
                        date_range = pd.date_range(start=start_date, periods=len(df), freq='MS')
                    
                    df['Order Date'] = date_range
                    st.sidebar.success(f"Generated {len(date_range)} {frequency.lower()} dates")
                
                elif date_option == "Use row index as time sequence":
                    # Use row index as sequential time
                    df['Order Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
                    st.sidebar.info("Using row order as time sequence (starting from 2023-01-01)")
                
                else:  # Skip time series
                    # Create a dummy date - we'll show aggregated analysis only
                    df['Order Date'] = pd.to_datetime('2024-01-01')
                    st.sidebar.info("Time series forecasting disabled - showing aggregated analysis only")
            
            # Sales column selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.sidebar.error("No numeric columns found in your dataset. Please upload a file with numeric sales/revenue data.")
                st.stop()
            
            if sales_col and sales_col in numeric_cols:
                default_sales_idx = numeric_cols.index(sales_col)
            else:
                default_sales_idx = 0
            
            sales_col = st.sidebar.selectbox(
                "Sales/Revenue Column",
                options=numeric_cols,
                index=default_sales_idx
            )
            
            df['Sales'] = pd.to_numeric(df[sales_col], errors='coerce')
            
            # Remove rows with invalid sales values
            df = df[df['Sales'].notna()].copy()
            
            if len(df) == 0:
                st.sidebar.error("No valid sales data found. Please check your sales column.")
                st.stop()
            
            # Optional categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Remove the date column if it's in categorical
            categorical_cols = [col for col in categorical_cols if col != date_col and df[col].nunique() < 100]
            
            if categorical_cols:
                st.sidebar.subheader("Optional Columns")
                st.sidebar.markdown("Add dimensions for deeper analysis:")
                
                category_options = ['None'] + categorical_cols
                
                category_col = st.sidebar.selectbox(
                    "Category Column (optional)",
                    options=category_options
                )
                
                if category_col != 'None':
                    df['Category'] = df[category_col].astype(str)
                else:
                    df['Category'] = 'All Products'
                
                segment_col = st.sidebar.selectbox(
                    "Segment Column (optional)",
                    options=category_options
                )
                
                if segment_col != 'None':
                    df['Segment'] = df[segment_col].astype(str)
                else:
                    df['Segment'] = 'All Customers'
                
                region_col = st.sidebar.selectbox(
                    "Region Column (optional)",
                    options=category_options
                )
                
                if region_col != 'None':
                    df['Region'] = df[region_col].astype(str)
                else:
                    df['Region'] = 'All Regions'
            else:
                df['Category'] = 'All Products'
                df['Segment'] = 'All Customers'
                df['Region'] = 'All Regions'
            
            # Add profit if not present
            if 'Profit' not in df.columns:
                profit_margin = st.sidebar.slider(
                    "Estimated Profit Margin (%)",
                    min_value=0,
                    max_value=100,
                    value=20,
                    help="Used to estimate profit since no profit column was found"
                )
                df['Profit'] = df['Sales'] * (profit_margin / 100)
            else:
                df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
            
            st.sidebar.success("Column mapping complete!")
            
            # Show data quality summary
            with st.sidebar.expander("Data Quality Summary"):
                st.write(f"Total rows: {len(df):,}")
                st.write(f"Date range: {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
                st.write(f"Categories: {df['Category'].nunique()}")
                st.write(f"Segments: {df['Segment'].nunique()}")
                st.write(f"Regions: {df['Region'].nunique()}")
    
    else:
        st.info("Please upload a file to begin")
        st.stop()

else:
    # Load sample dataset
    @st.cache_data
    def load_data():
        df = pd.read_csv('Sample - Superstore.csv', encoding='latin1')
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')
        return df
    
    df = load_data()

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
    default=df['Category'].unique()
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
filtered_df = df[
    (df['Category'].isin(category_filter)) & 
    (df['Segment'].isin(segment_filter)) &
    (df['Region'].isin(region_filter))
]

# ============= KEY METRICS SECTION =============
st.subheader("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

total_revenue = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
total_orders = len(filtered_df)
avg_order_value = filtered_df['Sales'].mean()
profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0

col1.metric("Total Revenue", f"${total_revenue:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Total Orders", f"{total_orders:,}")
col4.metric("Avg Order Value", f"${avg_order_value:.2f}")
col5.metric("Profit Margin", f"{profit_margin:.1f}%")

st.markdown("---")

# ============= FEATURE 2: FEATURE IMPORTANCE ANALYSIS =============
st.subheader("Feature Importance Analysis")
st.markdown("Understanding what drives your sales")

# Check if we have enough data and features for analysis
categorical_features = []
if 'Category' in filtered_df.columns and filtered_df['Category'].nunique() > 1:
    categorical_features.append('Category')
if 'Segment' in filtered_df.columns and filtered_df['Segment'].nunique() > 1:
    categorical_features.append('Segment')
if 'Region' in filtered_df.columns and filtered_df['Region'].nunique() > 1:
    categorical_features.append('Region')

# Add time-based features
time_features = ['Month', 'DayOfWeek', 'Quarter']

if len(categorical_features) > 0 or len(filtered_df) > 100:
    with st.spinner('Analyzing feature importance...'):
        # Prepare data
        df_model = filtered_df.copy()
        
        # Add time features
        df_model['Month'] = df_model['Order Date'].dt.month
        df_model['DayOfWeek'] = df_model['Order Date'].dt.dayofweek
        df_model['Quarter'] = df_model['Order Date'].dt.quarter
        df_model['Year'] = df_model['Order Date'].dt.year
        
        # Only add Year if there's variation
        if df_model['Year'].nunique() > 1:
            time_features.append('Year')
        
        # Encode categorical variables
        le_dict = {}
        encoded_features = []
        
        for feature in categorical_features:
            if feature in df_model.columns:
                le = LabelEncoder()
                df_model[f'{feature}_encoded'] = le.fit_transform(df_model[feature].astype(str))
                le_dict[feature] = le
                encoded_features.append(f'{feature}_encoded')
        
        # Combine all features
        feature_cols = encoded_features + time_features
        
        # Remove features with no variation
        feature_cols = [col for col in feature_cols if col in df_model.columns and df_model[col].nunique() > 1]
        
        if len(feature_cols) >= 2:  # Need at least 2 features for meaningful analysis
            X = df_model[feature_cols]
            y = df_model['Sales']
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            rf.fit(X, y)
            
            # Get feature importance
            importance_dict = {}
            for i, col in enumerate(feature_cols):
                # Map back to original feature names
                if col.endswith('_encoded'):
                    original_name = col.replace('_encoded', '')
                    importance_dict[original_name] = rf.feature_importances_[i]
                else:
                    importance_dict[col] = rf.feature_importances_[i]
            
            # Plot feature importance
            fig_importance = go.Figure()
            
            features = list(importance_dict.keys())
            importances = list(importance_dict.values())
            
            # Sort by importance
            sorted_indices = np.argsort(importances)[::-1]
            features_sorted = [features[i] for i in sorted_indices]
            importances_sorted = [importances[i] for i in sorted_indices]
            
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
                
                st.markdown(f"- Top 3 factors account for {sum(importances_sorted[:top_n])*100:.1f}% of sales variance")
        
        else:
            st.info("Not enough feature variation for importance analysis. Try adjusting your filters or ensure your dataset has categorical columns.")

else:
    st.info("""
    **Feature Importance Analysis requires:**
    - At least one categorical column (Category, Segment, Region, etc.), OR
    - A larger dataset (100+ records) for time-based analysis
    
    Your current dataset has limited features. Consider:
    - Including categorical columns in your data upload
    - Using a larger dataset
    - Ensuring categorical columns have multiple unique values
    """)

st.markdown("---")

# ============= HISTORICAL SALES WITH ANOMALY DETECTION =============
st.subheader("Historical Sales Analysis with Anomaly Detection")

df_daily = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
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
        
        if sales > row['rolling_mean']:
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
    
    # Show top 5 most recent anomalies
    recent_anomalies = anomalies.sort_values('Order Date', ascending=False).head(5)
    
    for _, row in recent_anomalies.iterrows():
        date = row['Order Date'].strftime('%Y-%m-%d')
        sales = row['Sales']
        z_score = row['z_score']
        
        if sales > row['rolling_mean']:
            alert_type = "Unusual Spike"
            color = "success"
        else:
            alert_type = "Unusual Drop"
            color = "warning"
        
        st.markdown(f"""
        <div class="anomaly-alert">
            <strong>{alert_type} detected on {date}</strong><br>
            Sales: ${sales:,.2f} (Z-score: {z_score:.2f})<br>
            <em>Investigate: Check for promotions, holidays, or data errors</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No significant anomalies detected in the current data filter")

st.markdown("---")

# ============= FORECASTING ENGINE =============
st.subheader("Sales Forecast with What-If Scenarios")

forecast_df = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
forecast_df.columns = ['ds', 'y']

@st.cache_resource
def train_model(data):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        interval_width=0.95
    )
    model.fit(data)
    return model

if len(forecast_df) > 30:
    with st.spinner('Training forecasting model...'):
        model = train_model(forecast_df)
    
    future = model.make_future_dataframe(periods=forecast_months * 30, freq='D')
    forecast = model.predict(future)
    
    forecast_future = forecast[forecast['ds'] > forecast_df['ds'].max()].copy()
    
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
    percent_change = ((scenario_total - base_total) / base_total) * 100 if base_total > 0 else 0
    
    col1.metric("Base Forecast", f"${base_total:,.0f}")
    col2.metric("Scenario Forecast", f"${scenario_total:,.0f}", f"{percent_change:+.1f}%")
    col3.metric("Revenue Impact", f"${revenue_diff:+,.0f}")
    col4.metric("Total Adjustment", f"{(total_impact - 1) * 100:+.1f}%")
    
    # ============= MONTE CARLO SIMULATION =============
    st.markdown("---")
    st.subheader("Monte Carlo Risk Analysis")
    
    @st.cache_data
    def run_monte_carlo(forecast_values, num_sims, uncertainty):
        np.random.seed(42)
        simulations = []
        for _ in range(num_sims):
            random_factors = np.random.normal(1.0, uncertainty/100, len(forecast_values))
            sim_result = forecast_values * random_factors
            simulations.append(sim_result)
        return np.array(simulations)
    
    with st.spinner(f'Running {num_simulations:,} Monte Carlo simulations...'):
        simulations = run_monte_carlo(
            forecast_future['yhat_scenario'].values,
            num_simulations,
            uncertainty_level
        )
    
    p10 = np.percentile(simulations, 10, axis=0)
    p25 = np.percentile(simulations, 25, axis=0)
    p50 = np.percentile(simulations, 50, axis=0)
    p75 = np.percentile(simulations, 75, axis=0)
    p90 = np.percentile(simulations, 90, axis=0)
    
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
    
    for percentile, value, color in [(10, total_p10, 'red'), (50, total_p50, 'green'), (90, total_p90, 'orange')]:
        fig_hist.add_vline(
            x=value, line_dash="dash", line_color=color,
            annotation_text= f"P{percentile}: ${value:,.0f}"
)

fig_hist.update_layout(
    title='Revenue Distribution',
    xaxis_title='Total Revenue ($)',
    yaxis_title='Frequency',
    showlegend=False,
    height=350
)

st.plotly_chart(fig_hist, use_container_width=True)

# ============= FEATURE 3: EXPORT & REPORTING =============
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

insights_text = f"""Top Sales Driver: {features_sorted[0]} ({importances_sorted[0]*100:.1f}% impact)
Anomalies Detected: {len(anomalies)}
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
        st.warning("Not enough data after filtering. Please adjust your filters.")
        st.markdown("---")
        st.markdown("Built with Streamlit, Prophet, and Plotly | Data updates in real-time as you adjust filters")

