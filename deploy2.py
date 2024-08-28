import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('gbr.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset for EDA
data = pd.read_csv('solarpowergeneration.csv')

# Streamlit app
st.title("Solar Power Generation Prediction")

st.sidebar.header("Input Features")
temp = st.sidebar.slider("Temperature (°C)", min_value=0, max_value=50, value=25)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=50)
windspeed = st.sidebar.slider("Wind Speed (m/s)", min_value=0, max_value=20, value=5)
precipitation = st.sidebar.slider("Precipitation (mm)", min_value=0, max_value=50, value=0)

# Predict button
if st.sidebar.button("Predict"):
    input_data = np.array([[temp, humidity, windspeed, precipitation]])
    prediction = model.predict(input_data)
    st.subheader(f"Predicted Solar Power Generation: {prediction[0]:.2f} MW")

# EDA Section
st.header("Exploratory Data Analysis")

st.subheader("Dataset Preview")
st.dataframe(data.head())

st.subheader("Statistical Summary")
st.write(data.describe())

st.subheader("Correlation Heatmap")
fig = px.imshow(data.corr(), text_auto=True, aspect="auto")
st.plotly_chart(fig)

st.subheader("Distribution of Solar Power Generation")
fig = px.histogram(data, x="Solar_Power_Generation", nbins=20)
st.plotly_chart(fig)
