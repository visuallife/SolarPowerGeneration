import pandas as pd
import streamlit as st
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check if the model files exist before proceeding
if not os.path.exists('solar_model.pkl') or not os.path.exists('scaler.pkl'):
    st.error("Error: The files 'solar_model.pkl' and/or 'scaler.pkl' were not found.")
    st.stop()

# Load the trained model and scaler
try:
    model = joblib.load('solar_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title("Solar Power Generation Predictor")
st.write("Enter the environmental conditions to predict the power output.")

# Define the features for the Streamlit UI
FEATURES = {
    'distance-to-solar-noon': 'Distance to Solar Noon',
    'temperature': 'Temperature (°F)',
    'wind-direction': 'Wind Direction (°)',
    'wind-speed': 'Wind Speed (mph)',
    'sky-cover': 'Sky Cover (oktas)',
    'visibility': 'Visibility (miles)',
    'humidity': 'Humidity (%)',
    'average-wind-speed-(period)': 'Avg. Wind Speed (mph)',
    'average-pressure-(period)': 'Avg. Pressure (inHg)'
}

# Create input widgets
input_data = {}
for key, label in FEATURES.items():
    input_data[key] = st.number_input(label, step=0.1)

# Prediction button
if st.button("Predict Power Generation"):
    # Convert input data to a DataFrame
    df_input = pd.DataFrame([input_data])

    # Scale the input data using the pre-trained scaler
    scaled_input = scaler.transform(df_input)

    # Make a prediction
    prediction = model.predict(scaled_input)
    predicted_power = round(prediction[0], 2)

    st.subheader("Predicted Power Generated")
    st.success(f"{predicted_power} Watts")
