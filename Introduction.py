import streamlit as st

# --- Page Configuration (Optional but Recommended) ---
st.set_page_config(
    page_title="Analysis Dashboard",
    page_icon="📊",  # You can use an emoji or a path to an image
    layout="wide"    # Can be "centered" or "wide"
)

# --- Main Introduction Page Content ---
st.title("Welcome to the Analysis Dashboard!")
st.subheader("Created by KVS Mohan Vamsi (st20314789)")

st.markdown("---") # Adds a horizontal rule for separation

st.header("Getting Started:")

st.markdown("""
Welcome to the data analysis dashboard made by KVS Mohan Vamsi (st20314789)

**Here's how to navigate the dashboard:**

1.  **Begin with Exploratory Data Analysis (EDA):**
    * Navigate to the **`EDA`** page using the side tabs.
    * On this page, you'll discover how the raw data is filtered, cleaned, and processed.
    * Explore visualizations and statistical summaries to uncover initial patterns, anomalies, and valuable insights hidden within dataset.

2.  **Train Predictive Models:**
    * Once EDA is done, head over to the **`Prediction Models`** tab.
    * Here, you can select from various machine learning algorithms and train them on the processed data.
    * The goal is to build models capable of predicting different factors based on the learned patterns.
    
""")

st.markdown("---")
st.info("Use the tabs on the left to switch between different sections of the dashboard.")