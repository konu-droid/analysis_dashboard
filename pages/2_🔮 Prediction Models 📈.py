import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib # To save/load models (optional)
import time
import plotly.graph_objs as go

# --- Configuration ---
st.set_page_config(page_title="Prediction Models", layout="wide")
st.title("🔮 Prediction Models 📈")
st.markdown("""
This page builds and evaluates machine learning models to predict air quality (e.g., PM2.5)
based on meteorological data and other pollutants.
""")

# --- Constants ---
DATA_DIR = "Weather_Dataset" # Relative path to the dataset directory
MODEL_DIR = "models" # Directory to save trained models
TARGET_VARIABLE = 'PM2.5' # The variable we want to predict

# --- Ensure model directory exists ---
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- Data Loading and Caching (Copied from EDA page for standalone use) ---
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
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            df.set_index('datetime', inplace=True)
            df.drop(['No', 'year', 'month', 'day', 'hour'], axis=1, inplace=True)
            df_list.append(df)
        except Exception as e:
            st.warning(f"Could not read or process file {file}: {e}")

    if not df_list:
        st.error("Error: No data could be loaded from the CSV files.")
        return None

    merged_df = pd.concat(df_list)
    merged_df.sort_index(inplace=True)
    return merged_df

# --- Preprocessing (Copied from EDA page) ---
def preprocess_data(df, target_variable):
    """
    Performs preprocessing steps like handling missing values.

    Args:
        df (pd.DataFrame): The raw merged dataframe.
        target_variable (str): The name of the target column.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    numeric_cols_to_fill = df.select_dtypes(include=np.number).columns
    df[numeric_cols_to_fill] = df[numeric_cols_to_fill].fillna(method='ffill')

    if 'wd' in df.columns:
        most_frequent_wd = df['wd'].mode()[0] if not df['wd'].mode().empty else 'Unknown'
        df['wd'].fillna(most_frequent_wd, inplace=True)

    if target_variable in df.columns:
        df.dropna(subset=[target_variable], inplace=True)
    else:
        st.error(f"Target variable '{target_variable}' not found in the dataframe columns.")
        return None

    # Drop station column if it exists, as it's categorical and might add too many features if one-hot encoded directly
    # A better approach might be target encoding or grouping stations. For simplicity, we drop it here.
    if 'station' in df.columns:
        df = df.drop('station', axis=1)

    return df

# --- Model Training Function ---
def train_model(df, target_variable, model_choice, test_size=0.2, random_state=42):
    """
    Prepares data, trains a selected model, and returns the pipeline and metrics.

    Args:
        df (pd.DataFrame): Preprocessed data.
        target_variable (str): Name of the target column.
        model_choice (str): Name of the model to train ('Linear Regression', 'Random Forest').
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (pipeline, X_test, y_test, metrics) or (None, None, None, None) if error.
    """
    try:
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]

        # Identify numerical and categorical features
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        # Create preprocessing pipelines for numerical and categorical features
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore')) # Ignore unknown categories in test set
        ])

        # Create a column transformer to apply different transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Define the model based on choice
        if model_choice == 'Linear Regression':
            model = LinearRegression()
        elif model_choice == 'Random Forest':
            # Use smaller n_estimators for faster training in demo
            model = RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1, max_depth=15, min_samples_split=10)
        else:
            st.error("Invalid model choice selected.")
            return None, None, None, None

        # Create the full pipeline: preprocess -> model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Train the model
        start_time = time.time()
        st.info(f"Training {model_choice} model...")
        pipeline.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        st.success(f"Model training completed in {training_time:.2f} seconds.")

        # Evaluate the model
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        metrics = {
            "R-squared (R2)": r2,
            "Mean Absolute Error (MAE)": mae,
            "Mean Squared Error (MSE)": mse,
            "Root Mean Squared Error (RMSE)": rmse,
            "Training Time (s)": training_time
        }

        return pipeline, X_test, y_test, metrics

    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

# --- Load and Prepare Data ---
raw_df = load_and_merge_data(DATA_DIR)

if raw_df is not None:
    st.success(f"Successfully loaded and merged data from {len(glob.glob(os.path.join(DATA_DIR, 'PRSA_Data_*.csv')))} stations.")

    processed_df = preprocess_data(raw_df.copy(), TARGET_VARIABLE)

    if processed_df is not None:
        st.subheader("1. Model Selection and Training")

        # Select features (implicitly all columns except target after preprocessing)
        st.write(f"Target Variable: **{TARGET_VARIABLE}**")
        st.write("Features used for prediction (after preprocessing):")
        st.write(list(processed_df.drop(TARGET_VARIABLE, axis=1).columns))

        # Model selection dropdown
        model_options = ['Linear Regression', 'Random Forest']
        selected_model = st.selectbox("Choose a model to train:", model_options)

        # Train button
        if st.button(f"Train {selected_model} Model"):
            pipeline, X_test, y_test, metrics = train_model(processed_df, TARGET_VARIABLE, selected_model)

            if pipeline and metrics:
                st.session_state['pipeline'] = pipeline # Store pipeline in session state
                st.session_state['metrics'] = metrics
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['model_name'] = selected_model

                # Save the trained model (optional)
                model_filename = os.path.join(MODEL_DIR, f"{selected_model.replace(' ', '_').lower()}_pipeline.joblib")
                try:
                    joblib.dump(pipeline, model_filename)
                    st.success(f"Trained model pipeline saved to {model_filename}")
                except Exception as e:
                    st.warning(f"Could not save the model: {e}")

            else:
                st.error("Model training failed. Check logs for details.")

        # --- Display Results if Model Trained ---
        if 'pipeline' in st.session_state and 'metrics' in st.session_state:
            st.subheader("2. Model Evaluation")
            st.write(f"Results for **{st.session_state['model_name']}** model:")

            # Display metrics
            st.write("Performance Metrics on Test Set:")
            metrics_df = pd.DataFrame([st.session_state['metrics']])
            st.dataframe(metrics_df)

            # Plot actual vs predicted
            st.write("Actual vs. Predicted Values (Test Set)")
            y_pred = st.session_state['pipeline'].predict(st.session_state['X_test'])
            results_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': y_pred})

            # Sample for plotting if large
            plot_df = results_df.sample(min(1000, len(results_df)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df['Actual'], y=plot_df['Predicted'], mode='markers', name='Predictions', marker=dict(opacity=0.6)))
            fig.add_trace(go.Scatter(x=[plot_df['Actual'].min(), plot_df['Actual'].max()], y=[plot_df['Actual'].min(), plot_df['Actual'].max()], mode='lines', name='Ideal Fit', line=dict(dash='dash', color='red')))
            fig.update_layout(title="Actual vs. Predicted PM2.5", xaxis_title="Actual PM2.5", yaxis_title="Predicted PM2.5", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # --- Section for Making Predictions (Optional) ---
            st.subheader("3. Make a Prediction")
            st.write("Enter values for the features to predict PM2.5 (using the trained model):")

            # Create input fields dynamically based on features
            input_data = {}
            num_cols = 3 # Number of columns for input fields
            cols = st.columns(num_cols)
            col_idx = 0

            # Get feature names from the pipeline's preprocessor step
            try:
                # Access transformers from ColumnTransformer
                transformers = st.session_state['pipeline'].named_steps['preprocessor'].transformers_
                feature_names_num = transformers[0][2] # Numerical features
                feature_names_cat = transformers[1][2] # Categorical features

                all_feature_names = list(feature_names_num) + list(feature_names_cat)

                # Get unique categories for 'wd' if it's a feature
                wd_options = list(processed_df['wd'].unique()) if 'wd' in feature_names_cat else []

                for feature in all_feature_names:
                    with cols[col_idx % num_cols]:
                        if feature in feature_names_num:
                            # Use mean as default for numerical inputs
                            default_val = float(processed_df[feature].mean())
                            input_data[feature] = st.number_input(f"Enter {feature}", value=default_val, format="%.2f")
                        elif feature == 'wd' and wd_options:
                            input_data[feature] = st.selectbox(f"Select {feature}", options=wd_options, index=0)
                        elif feature in feature_names_cat:
                             # Simple text input for other potential categorical vars (if any)
                             default_cat_val = processed_df[feature].mode()[0] if not processed_df[feature].mode().empty else ""
                             input_data[feature] = st.text_input(f"Enter {feature}", value=default_cat_val)
                    col_idx += 1

                if st.button("Predict PM2.5"):
                    input_df = pd.DataFrame([input_data])
                    # Ensure column order matches training data before prediction
                    input_df = input_df[all_feature_names]

                    try:
                        prediction = st.session_state['pipeline'].predict(input_df)
                        st.success(f"Predicted {TARGET_VARIABLE}: **{prediction[0]:.2f}**")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        st.error("Ensure input values are appropriate.")

            except Exception as e:
                st.error(f"Could not retrieve feature names for prediction input: {e}")


    else:
        st.warning("Data preprocessing failed. Cannot proceed with modeling.")

else:
    st.warning("Data could not be loaded. Please check the file paths and data integrity.")


st.sidebar.header("About")
st.sidebar.info(f"This page trains and evaluates models to predict {TARGET_VARIABLE}.")
