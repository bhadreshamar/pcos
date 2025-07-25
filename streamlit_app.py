import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
  data = pd.read_csv("Data/PCOS_data_infertility.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data

st.write ('PCOS')

st.info ('This app builds a machine learning model to detect PCOS')

