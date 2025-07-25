import streamlit as st
import pickle
import pandas as pd


def get_clean_data():
  data = pd.read_csv("Data/PCOS_data_infertility.csv")
  
  data = data.drop(['Unnamed: 44', 'Si. No', 'Patient File No.'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


# Add a title
st.set_page_config(page_title="Detect PCOS",
                    page_icon="👩‍⚕️", 
                    layout="wide", 
                    initial_sidebar_state="expanded")

# Set up the structure
with st.container():
    st.title("Breast Cancer Diagnosis")
    st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")


st.write ('PCOS')

st.info ('This app builds a machine learning model to detect PCOS')

