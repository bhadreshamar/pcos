import streamlit as st
import pickle
import pandas as pd


def get_clean_data():
  data = pd.read_csv("Data/PCOS_data_infertility.csv")
  
  data = data.drop(['Unnamed: 44', 'Si. No', 'Patient File No.'], axis=1)
  
  data['PCOS (Y/N)'] = data['PCOS (Y/N)'].map({ 'Y': 1, 'N': 0 })
  
  return data


# Add a title
st.set_page_config(page_title="Detect PCOS",
                    page_icon="👩‍⚕️", 
                    layout="wide", 
                    initial_sidebar_state="expanded")

# Set up the structure
with st.container():
    st.title("PCOS Disgnosis")
    st.write("Please input current medical history and symptoms. This app predicts PCOS diagnosis using Machine Learning ")
    col1, col2 = st.columns([4,1])
    with col1:
        st.write("Column 1")
    with col2:
        st.write("Column 2")
        


