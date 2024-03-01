import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="FloodWatch", page_icon=":bar_chart:",layout="wide", initial_sidebar_state='expanded')
st.title(" :bar_chart:  FloodWatch : An AutoML tool for flood forecasting and area segmentation and classification using weather data and images")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.sidebar.write("Bangladesh is prone to frequent floods due to its low-lying location and vulnerability to monsoon rains, melting snow from the Himalayas, deforestation, and local rainfall. This Streamlit interface is designed to promote awareness among users and empower them to analyse various flood-related problems using machine learning techniques through meteorological, tabular and image data.")

dataframe = pd.read_csv('./FloodPrediction_Preprocessed.csv')
data = dataframe.groupby(['Station_Names','Year']).aggregate({'Station_Names':'first', 'Year':'first', 'LATITUDE':'first', 'LONGITUDE':'first', 'Flood?':'sum', 'Rainfall':'sum'})

fig = px.scatter_mapbox(data, lat='LATITUDE', lon='LONGITUDE', color='Rainfall', range_color=[0,6000], hover_name='Station_Names', mapbox_style='carto-positron', height=800, width=700, size='Flood?', color_continuous_scale='reds', size_max=30, title='Annual Flood Frequency Around Bangladesh', zoom=6, animation_frame='Year')
st.plotly_chart(fig)