import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # We'll use this carefully for time series
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # To save/load models or preprocessors if needed

st.set_page_config(layout="wide")
st.title("🏭 Air Quality Prediction Model Building")

# --- Configuration & Constants ---
# Make sure this path is correct relative to this script's location
# Or provide an absolute path
dataset_path = "Weather_Dataset"
# Define expected columns (SYNC WITH EDA PAGE / ACTUAL DATA)
EXPECTED_POLLUTANTS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
EXPECTED_METEO = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
EXPECTED_TIME = ['year', 'month', 'day', 'hour']
EXPECTED_STATION = 'station'

# --- Data Loading and Caching ---
# Reusing the loading logic concept from the EDA page
# Cache the data loading and initial processing
@st.cache_data
def load_and_prepare_data(folder_path):
    """Loads, merges, and performs initial preparation."""
    try:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not csv_files:
            st.error(f"No CSV files found in: {folder_path}")
            return None
        dataframes = []
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                if EXPECTED_STATION not in df.columns:
                    station_name = file.split('_')[0]
                    df[EXPECTED_STATION] = station_name
                dataframes.append(df)
            except Exception as e:
                st.warning(f"Skipping {file} due to error: {e}")
                continue
        if not dataframes:
            st.error("No data loaded.")
            return None

        merged_df = pd.concat(dataframes, ignore_index=True)
        st.success(f"Loaded data from {len(dataframes)} files ({len(merged_df)} rows).")
        
        # --- Basic Preprocessing ---
        # 1. Create Datetime (Crucial for time series split and features)
        required_cols_for_datetime = EXPECTED_TIME
        if not all(col in merged_df.columns for col in required_cols_for_datetime):
            st.error(f"Missing required time columns for datetime creation: {required_cols_for_datetime}")
            return None
        try:
            merged_df['datetime'] = pd.to_datetime(merged_df[required_cols_for_datetime], errors='coerce')
            merged_df.dropna(subset=['datetime'], inplace=True) # Drop rows where datetime failed
            merged_df.set_index('datetime', inplace=True)
            merged_df.sort_index(inplace=True) # Essential for time series
            st.write("Created and set 'datetime' index.")
        except Exception as e:
            st.error(f"Error creating datetime index: {e}")
            return None
            
        # 2. Handle Duplicates (based on station and new datetime index)
        merged_df.reset_index(inplace=True) # Temp reset for duplicate check
        merged_df.sort_values(by=[EXPECTED_STATION, 'datetime'], inplace=True)
        duplicates = merged_df.duplicated(subset=[EXPECTED_STATION, 'datetime'], keep='first')
        num_duplicates = duplicates.sum()
        if num_duplicates > 0:
            merged_df = merged_df[~duplicates]
            st.write(f"Removed {num_duplicates} duplicate entries (same station/timestamp).")
        merged_df.set_index('datetime', inplace=True) # Set index back
            
        # 3. Imputation (Example: Linear interpolation per station for pollutants/meteo, 0 for RAIN)
        # Note: Imputation *before* feature engineering like time features
        numeric_cols = EXPECTED_POLLUTANTS + EXPECTED_METEO
        numeric_cols_present = [col for col in numeric_cols if col in merged_df.columns]
        numeric_cols_to_interpolate = [col for col in numeric_cols_present if col != 'RAIN']

        if numeric_cols_to_interpolate:
            st.write(f"Interpolating missing values for: {numeric_cols_to_interpolate}")
            # Ensure index is sorted before group-wise interpolation
            merged_df.sort_index(inplace=True)
            merged_df[numeric_cols_to_interpolate] = merged_df.groupby(EXPECTED_STATION)[numeric_cols_to_interpolate].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both', axis=0)
            )
        if 'RAIN' in merged_df.columns:
            merged_df['RAIN'].fillna(0, inplace=True)
            st.write("Filled missing 'RAIN' with 0.")
            
        # Drop rows where target might still be NaN after interpolation (important!)
        target_options = [col for col in EXPECTED_POLLUTANTS if col in merged_df.columns]
        if target_options: # Check if any target columns exist
           initial_rows = len(merged_df)
           merged_df.dropna(subset=target_options, inplace=True)
           rows_dropped = initial_rows - len(merged_df)
           if rows_dropped > 0:
               st.warning(f"Dropped {rows_dropped} rows with remaining NaN values in potential target pollutant columns.")
        
        st.success("Initial data preparation complete.")
        return merged_df

    except FileNotFoundError:
        st.error(f"Dataset directory not found: {folder_path}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during loading/preparation: {e}")
        return None

# --- Model Building Functions ---

def feature_engineering(df):
    """Creates time-based features."""
    st.write("Performing Feature Engineering...")
    df_eng = df.copy()
    
    # Ensure index is datetime
    if not isinstance(df_eng.index, pd.DatetimeIndex):
         st.error("Feature engineering requires a DatetimeIndex. Please ensure data preparation was successful.")
         return None

    # Create time-based features
    df_eng['hour_of_day'] = df_eng.index.hour
    df_eng['day_of_week'] = df_eng.index.dayofweek # Monday=0, Sunday=6
    df_eng['month'] = df_eng.index.month
    # Optional: Add cyclical features for time (better for linear models)
    # df_eng['hour_sin'] = np.sin(2 * np.pi * df_eng['hour_of_day']/24)
    # df_eng['hour_cos'] = np.cos(2 * np.pi * df_eng['hour_of_day']/24)
    st.write("Created features: 'hour_of_day', 'day_of_week', 'month'.")

    # **Advanced Idea (Lag Features):**
    # Lag features are VERY important for time series forecasting.
    # Example: Create lagged PM2.5 for 1 and 3 hours ago FOR EACH station
    # target = 'PM2.5' # Example
    # if target in df_eng.columns:
    #     df_eng[f'{target}_lag1'] = df_eng.groupby(EXPECTED_STATION)[target].shift(1)
    #     df_eng[f'{target}_lag3'] = df_eng.groupby(EXPECTED_STATION)[target].shift(3)
    #     # Important: Need to handle NaNs introduced by shift (e.g., dropna or careful imputation)
    #     st.write("Created example lag features (requires handling NaNs).")
    # For simplicity in this initial version, we are not using lag features by default.

    return df_eng

def split_data_temporal(df, test_size=0.2):
    """Splits data temporally based on the datetime index."""
    st.write(f"Splitting data temporally: Test size = {test_size*100:.0f}%")
    
    # Ensure the data is sorted by time
    df_sorted = df.sort_index()
    
    split_index = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]
    
    st.write(f"Training data shape: {train_df.shape} (From {train_df.index.min()} to {train_df.index.max()})")
    st.write(f"Test data shape: {test_df.shape} (From {test_df.index.min()} to {test_df.index.max()})")
    
    if len(train_df) == 0 or len(test_df) == 0:
        st.error("Data splitting resulted in empty train or test set. Check data range and test size.")
        return None, None, None, None
        
    return train_df, test_df

def build_preprocessor(numeric_features, categorical_features):
    """Builds a ColumnTransformer for preprocessing."""
    
    transformers = []
    
    # Numeric features: Imputation (already done) + Scaling
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()) # Scale numeric features
        ])
        transformers.append(('num', numeric_transformer, numeric_features))
        st.write(f"Applying StandardScaler to: {numeric_features}")
        
    # Categorical features: Imputation (if needed) + One-Hot Encoding
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Convert categories to numbers
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
        st.write(f"Applying OneHotEncoder to: {categorical_features}")

    if not transformers:
         st.error("No features selected for preprocessing.")
         return None

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough') # Keep other columns if any
    return preprocessor


def train_evaluate_model(train_df, test_df, target_col, features, model_choice='RandomForest'):
    """Builds pipeline, trains model, evaluates, and returns results."""
    
    st.write(f"--- Training Model: {model_choice} ---")
    st.write(f"Target Variable: {target_col}")
    st.write(f"Features Used: {features}")
    
    # Separate features (X) and target (y)
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]
    
    # Identify feature types for preprocessor
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()
    
    # Remove target from numeric features if accidentally included
    if target_col in numeric_features:
        numeric_features.remove(target_col)
        
    st.write(f"Identified Numeric Features for Scaling: {numeric_features}")
    st.write(f"Identified Categorical Features for Encoding: {categorical_features}")

    # --- Build Preprocessor ---
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    if preprocessor is None:
        return None, None, None # Stop if preprocessing failed

    # --- Define Model ---
    if model_choice == 'RandomForest':
        # Simple RF model - parameters can be tuned
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=10, min_samples_leaf=5) 
        st.write("Using RandomForestRegressor (with basic parameters). Consider hyperparameter tuning for better performance.")
    # Add other models here (e.g., LinearRegression, XGBoost)
    # elif model_choice == 'LinearRegression':
    #     from sklearn.linear_model import LinearRegression
    #     model = LinearRegression()
    else:
        st.error(f"Model choice '{model_choice}' not implemented.")
        return None, None, None

    # --- Create Full Pipeline ---
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    # --- Train the Model ---
    st.write("Training the pipeline...")
    pipeline.fit(X_train, y_train)
    st.success("Pipeline training complete.")

    # --- Make Predictions ---
    st.write("Making predictions on the test set...")
    y_pred = pipeline.predict(X_test)

    # --- Evaluate the Model ---
    st.subheader("Model Evaluation Metrics")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.3f}")
    col2.metric("MSE", f"{mse:.3f}")
    col3.metric("RMSE", f"{rmse:.3f}")
    col4.metric("R² Score", f"{r2:.3f}")
    st.write("Lower MAE, MSE, RMSE are better. R² closer to 1 is better.")

    # --- Visualizations ---
    st.subheader("Evaluation Visualizations")
    
    # 1. Predictions vs Actual
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=2, color='red') # Line y=x
    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title("Predicted vs. Actual Values")
    st.pyplot(fig1)
    plt.close(fig1)
    st.write("Points closer to the red dashed line indicate better predictions.")

    # 2. Feature Importances (if available)
    if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
        st.subheader("Feature Importances")
        try:
            # Get feature names after one-hot encoding
            feature_names = []
            ct_transformers = pipeline.named_steps['preprocessor'].transformers_
            
            for name, transformer, features in ct_transformers:
                if name == 'num':
                    feature_names.extend(features)
                elif name == 'cat' and hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                    feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(features))
                elif name == 'cat': # Fallback if get_feature_names_out not available
                    # This part might need adjustment depending on sklearn version and encoder type
                    st.warning("Could not reliably get encoded feature names for importance plot.")
            
            if feature_names:        
                importances = pipeline.named_steps['regressor'].feature_importances_
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20) # Show top 20

                fig2, ax2 = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax2)
                ax2.set_title('Top 20 Feature Importances')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.warning("Could not extract feature names to display feature importances.")

        except Exception as e:
            st.error(f"Error displaying feature importances: {e}")
            
    return pipeline, y_test, y_pred


# --- Streamlit App Main Logic ---

# Load data once
if 'prepared_data' not in st.session_state:
    st.session_state.prepared_data = None

if st.session_state.prepared_data is None:
     with st.spinner("Loading and preparing data... This might take a moment."):
        st.session_state.prepared_data = load_and_prepare_data(dataset_path)

if st.session_state.prepared_data is not None:
    df_prepared = st.session_state.prepared_data
    
    st.sidebar.header("Model Configuration")
    
    # --- User Inputs ---
    
    # 1. Select Target Variable
    possible_targets = [col for col in EXPECTED_POLLUTANTS if col in df_prepared.columns]
    target_col = st.sidebar.selectbox("Select Target Variable (Pollutant to Predict):", options=possible_targets, index=possible_targets.index('PM2.5') if 'PM2.5' in possible_targets else 0)
    
    # 2. Feature Engineering Step
    perform_fe = st.sidebar.checkbox("Perform Feature Engineering (Time Features)?", value=True)
    if perform_fe:
        with st.spinner("Running feature engineering..."):
            df_model_input = feature_engineering(df_prepared)
    else:
        df_model_input = df_prepared.copy() # Use prepared data directly

    if df_model_input is not None:
        # 3. Select Features
        st.sidebar.subheader("Feature Selection")
        # Default features: Meteo + Station + Time Features (if created)
        default_features = []
        meteo_available = [col for col in EXPECTED_METEO if col in df_model_input.columns]
        default_features.extend(meteo_available)
        if EXPECTED_STATION in df_model_input.columns:
             default_features.append(EXPECTED_STATION)
        
        time_features_created = ['hour_of_day', 'day_of_week', 'month']
        time_features_available = [col for col in time_features_created if col in df_model_input.columns]
        default_features.extend(time_features_available)

        # Optional: Allow including other pollutants as features
        include_other_pollutants = st.sidebar.checkbox("Include other pollutants as features?", value=False)
        other_pollutants = [p for p in possible_targets if p != target_col and p in df_model_input.columns]
        if include_other_pollutants and other_pollutants:
            default_features.extend(other_pollutants)
            
        # Ensure no duplicates and target is not in features
        default_features = sorted(list(set(default_features)))
        if target_col in default_features:
            default_features.remove(target_col)
            
        # Multiselect for features
        all_possible_features = [col for col in df_model_input.columns if col != target_col and col not in EXPECTED_TIME] # Exclude raw time cols
        selected_features = st.sidebar.multiselect(
            "Select Features for the Model:", 
            options=all_possible_features, 
            default=default_features
        )

        # 4. Select Model
        model_choice = st.sidebar.selectbox("Select Model:", options=['RandomForest']) # Add more later e.g., 'LinearRegression', 'XGBoost'
        
        # 5. Test Set Size
        test_size = st.sidebar.slider("Select Test Set Size (Temporal Split):", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

        # --- Train Button ---
        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Train and Evaluate Model", key="train_eval"):
            if not selected_features:
                 st.error("Please select at least one feature.")
            elif target_col in selected_features:
                 st.error(f"Target variable '{target_col}' cannot be included in the features.")
            else:
                # Check if all selected features exist
                missing_features = [f for f in selected_features if f not in df_model_input.columns]
                if missing_features:
                    st.error(f"Selected features not found in the data: {missing_features}")
                else:
                    with st.spinner(f"Training {model_choice} model..."):
                        # Split data
                        train_df, test_df = split_data_temporal(df_model_input, test_size=test_size)
                        
                        if train_df is not None and test_df is not None:
                             # Train and evaluate
                             pipeline, y_test, y_pred = train_evaluate_model(
                                 train_df, test_df, target_col, selected_features, model_choice
                             )
                             
                             # Store results if needed (e.g., for later comparison)
                             if pipeline:
                                st.session_state.last_pipeline = pipeline
                                st.session_state.last_y_test = y_test
                                st.session_state.last_y_pred = y_pred
                                st.success("Model training and evaluation finished.")
                        else:
                            st.error("Failed to split data. Cannot proceed with training.")
        else:
            st.info("Configure the model settings in the sidebar and click 'Train and Evaluate Model'.")

    else:
        st.error("Feature engineering failed. Cannot proceed with model building.")

else:
    st.warning("Data not loaded or prepared. Please check the data loading step.")