import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
st.set_page_config(page_title="EDA Insights", layout="wide")
st.title("🔍 Exploratory Data Analysis (EDA) Insights 📊")
st.markdown("""
This page performs Exploratory Data Analysis on the merged Beijing Air Quality dataset (2013-2017).
It covers data loading, preprocessing, basic statistics, and various visualizations.
""")

# --- Constants ---
DATA_DIR = "Weather_Dataset" # Relative path to the dataset directory

# --- Data Loading and Caching ---
def load_and_merge_data(data_dir):
    """
    Loads all CSV files from the specified directory, merges them,
    and performs initial datetime conversion.

    Args:
        data_dir (str): The path to the directory containing the CSV files.

    Returns:
        pd.DataFrame: A merged DataFrame containing data from all stations,
                      or None if the directory is not found or no CSV files are present.
    """
    if not os.path.isdir(data_dir):
        st.error(f"Error: Data directory not found at '{data_dir}'. Please ensure the 'Weather_Dataset' folder is in the same directory as the Streamlit app.")
        return None

    csv_files = glob.glob(os.path.join(data_dir, "PRSA_Data_*.csv"))

    if not csv_files:
        st.error(f"Error: No 'PRSA_Data_*.csv' files found in the directory '{data_dir}'.")
        return None

    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Combine year, month, day, hour into a single datetime column
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            # Drop original date/time columns
            df.drop(['No', 'year', 'month', 'day', 'hour'], axis=1, inplace=True)
            df_list.append(df)
        except Exception as e:
            st.warning(f"Could not read or process file {file}: {e}")

    if not df_list:
        st.error("Error: No data could be loaded from the CSV files.")
        return None

    # Merge all dataframes
    merged_df = pd.concat(df_list)
    merged_df.sort_index(inplace=True) # Sort by datetime index
    return merged_df

# --- Preprocessing ---
def preprocess_data(df):
    """
    Performs preprocessing steps like handling missing values.

    Args:
        df (pd.DataFrame): The raw merged dataframe.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # --- Handle Missing Values ---
    # Calculate percentage of missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_info = pd.DataFrame({'Missing Count': df.isnull().sum(), 'Missing Percentage': missing_percentage})

    # Simple forward fill for time-series data. More sophisticated methods could be used.
    # Identify numeric columns suitable for ffill (excluding categorical 'wd')
    numeric_cols_to_fill = df.select_dtypes(include=np.number).columns
    df[numeric_cols_to_fill] = df[numeric_cols_to_fill].fillna(method='ffill')

    # For wind direction ('wd'), fill NA with the most frequent value or 'Unknown'
    if 'wd' in df.columns:
        most_frequent_wd = df['wd'].mode()[0] if not df['wd'].mode().empty else 'Unknown'
        df['wd'].fillna(most_frequent_wd, inplace=True) # Or use ffill if preferred

    # Drop rows where the target variable (e.g., PM2.5) might still be NA after ffill (e.g., at the very beginning)
    # Important: Choose your primary target variable here if applicable
    target_var = 'PM2.5' # Example target
    if target_var in df.columns:
        df.dropna(subset=[target_var], inplace=True)

    return df, missing_info

# --- Load and Preprocess Data ---
raw_df = load_and_merge_data(DATA_DIR)

if raw_df is not None:
    st.success(f"Successfully loaded and merged data from {len(glob.glob(os.path.join(DATA_DIR, 'PRSA_Data_*.csv')))} stations.")

    processed_df, missing_info_initial = preprocess_data(raw_df.copy()) # Use copy to avoid modifying cached raw_df

    st.subheader("1. Data Overview & Preprocessing")

    # Display basic info
    st.markdown("#### Basic Information (Raw Data)")
    st.write("Shape of Raw Merged Data:", raw_df.shape)
    st.write("Data Types (Raw):")
    st.dataframe(raw_df.dtypes.astype(str))

    # Display missing values info before handling
    st.markdown("#### Missing Values (Before Preprocessing)")
    st.dataframe(missing_info_initial[missing_info_initial['Missing Count'] > 0])

    # Display info after preprocessing
    st.markdown("#### Basic Information (Processed Data)")
    st.write("Shape after Preprocessing (Missing Value Handling):", processed_df.shape)
    st.write("Missing Values After Preprocessing:", processed_df.isnull().sum().sum()) # Should be 0 or close

    # Display sample data
    st.markdown("#### Sample Data (Processed)")
    st.dataframe(processed_df.head())

    st.subheader("2. Descriptive Statistics")
    st.markdown("Summary statistics for numerical features:")
    st.dataframe(processed_df.describe())

    st.subheader("3. Univariate Analysis")
    st.markdown("Distribution of key pollutants and meteorological variables.")

    # Select columns for histograms
    numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
    # Exclude highly skewed columns like 'RAIN' if they dominate the plot scale too much initially
    cols_to_plot = [col for col in numeric_cols if col not in ['RAIN']] # Adjust as needed
    selected_col_hist = st.selectbox("Select variable for histogram:", options=cols_to_plot, index=cols_to_plot.index('PM2.5') if 'PM2.5' in cols_to_plot else 0)

    if selected_col_hist:
        fig_hist = px.histogram(processed_df, x=selected_col_hist, nbins=50, title=f"Distribution of {selected_col_hist}", marginal="box")
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Distribution of Wind Direction (Categorical)
    if 'wd' in processed_df.columns:
        st.markdown("#### Wind Direction Distribution")
        wd_counts = processed_df['wd'].value_counts().reset_index()
        wd_counts.columns = ['Wind Direction', 'Count']
        fig_wd = px.bar(wd_counts, x='Wind Direction', y='Count', title="Frequency of Wind Directions")
        st.plotly_chart(fig_wd, use_container_width=True)


    st.subheader("4. Bivariate Analysis")
    st.markdown("Relationships between pairs of variables.")

    # Scatter plots
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis for scatter plot:", options=numeric_cols, index=numeric_cols.index('WSPM') if 'WSPM' in numeric_cols else 0)
    with col2:
        y_axis = st.selectbox("Select Y-axis for scatter plot:", options=numeric_cols, index=numeric_cols.index('PM2.5') if 'PM2.5' in numeric_cols else 0)

    if x_axis and y_axis:
        # Sample data for performance if dataset is very large
        sample_df = processed_df.sample(min(5000, len(processed_df))) # Plot max 5000 points
        fig_scatter = px.scatter(sample_df, x=x_axis, y=y_axis, title=f"{y_axis} vs. {x_axis}", opacity=0.5)
        st.plotly_chart(fig_scatter, use_container_width=True)


    st.subheader("5. Multivariate Analysis")
    st.markdown("Correlation between numerical variables.")

    # Correlation Heatmap
    # Calculate correlation matrix only on numeric columns
    numeric_df = processed_df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()

    fig_heatmap, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    plt.title("Correlation Matrix of Numerical Features", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig_heatmap)


    st.subheader("6. Temporal Analysis")
    st.markdown("Trends over time.")

    # Allow user to select variable and time frequency
    temporal_var = st.selectbox("Select variable for time series plot:", options=numeric_cols, index=numeric_cols.index('PM2.5') if 'PM2.5' in numeric_cols else 0)
    resample_freq = st.selectbox("Select time frequency for aggregation:", options=['H', 'D', 'W', 'M'], format_func=lambda x: {'H':'Hourly (Raw)', 'D':'Daily Mean', 'W':'Weekly Mean', 'M':'Monthly Mean'}[x], index=1) # Default to Daily

    if temporal_var and resample_freq:
        # Resample data
        resampled_data = processed_df[temporal_var].resample(resample_freq).mean()

        fig_time = px.line(resampled_data, x=resampled_data.index, y=temporal_var, title=f"{temporal_var} Trend ({resample_freq} Average)")
        fig_time.update_xaxes(title_text='Date')
        fig_time.update_yaxes(title_text=f"Average {temporal_var}")
        st.plotly_chart(fig_time, use_container_width=True)

    # Seasonal analysis (e.g., average PM2.5 by month)
    st.markdown("#### Monthly Average Pattern")
    monthly_avg = processed_df.groupby(processed_df.index.month)[numeric_cols].mean()
    monthly_avg.index.name = 'Month'
    selected_var_monthly = st.selectbox("Select variable for monthly average plot:", options=numeric_cols, index=numeric_cols.index('PM2.5') if 'PM2.5' in numeric_cols else 0)

    if selected_var_monthly:
        fig_monthly = px.line(monthly_avg, x=monthly_avg.index, y=selected_var_monthly, title=f"Average {selected_var_monthly} by Month")
        fig_monthly.update_xaxes(title_text='Month of Year', dtick=1)
        st.plotly_chart(fig_monthly, use_container_width=True)

else:
    st.warning("Data could not be loaded. Please check the file paths and data integrity.")

st.sidebar.header("About")
st.sidebar.info("This page provides Exploratory Data Analysis for the Beijing Air Quality dataset.")

