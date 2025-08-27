import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("solar_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Solar Power Prediction", layout="centered")

st.title("☀️ Solar Power Generation Prediction")
st.write("Enter environmental conditions to predict solar power output (kW).")

def get_float_input(label, default="0.0"):
    val = st.text_input(label, default)
    try:
        return float(val)
    except ValueError:
        st.warning(f"⚠️ Please enter a valid number for {label}")
        return 0.0   # fallback allows 0

# 🔹 Input fields (all allow unlimited decimals & 0)
distance_noon   = get_float_input("Distance to Solar Noon", "0.0")
temperature     = get_float_input("Temperature (°C)", "0.0")
wind_direction  = get_float_input("Wind Direction (°)", "0.0")
wind_speed      = get_float_input("Wind Speed (m/s)", "0.0")
sky_cover       = get_float_input("Sky Cover (%)", "0.0")
visibility      = get_float_input("Visibility (km)", "0.0")
humidity        = get_float_input("Humidity (%)", "0.0")
avg_wind_speed  = get_float_input("Average Wind Speed (period)", "0.0")
avg_pressure    = get_float_input("Average Pressure (period)", "0.0")

# Collect inputs
input_data = np.array([[distance_noon, temperature, wind_direction, wind_speed,
                        sky_cover, visibility, humidity, avg_wind_speed, avg_pressure]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("🔮 Predict Power Generation"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"⚡ Predicted Power Generation: {prediction:.2f} kW")
