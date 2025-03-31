import os
import pandas as pd
import streamlit as st

# Folder containing CSV files
# folder_path = "C:\\Users\\USER\\Documents\\cardiffmet_robotics\\data_analysis\\analysis_dashboard\\Weather_Dataset"
dataset_path = "Weather_Dataset"

def load_csv_files():
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    
    # List to store individual DataFrames
    dataframes = []
    
    # Read each CSV file and append it to the list
    for file in csv_files:
        file_path = os.path.join(dataset_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    # Merge all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    st.success(f"Loaded all the files from {dataset_path} into dataframe")

    return merged_df

# Streamlit UI
st.title("EDA Insights")

if st.button("Load CSV Files"):
    dataset_df = load_csv_files()
    st.write(dataset_df.head())
    
    # Save merged data to a new CSV file
    # output_file = os.path.join(folder_path, "merged_weather_data.csv")
    # merged_df.to_csv(output_file, index=False)
    # st.success(f"Merged CSV saved to: {output_file}")