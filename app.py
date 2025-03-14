import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("Main Menu", ["EDA", 'Prediction Models'], 
        icons=['house', 'cloud-upload'], menu_icon="cast", default_index=1)
    selected