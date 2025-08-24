import pandas as pd
from flask import Flask, request, render_template, jsonify
import joblib
import os

# Check if the model files exist before proceeding
if not os.path.exists('solar_model.pkl') or not os.path.exists('scaler.pkl'):
    print("Error: The files 'solar_model.pkl' and/or 'scaler.pkl' were not found.")
    print("Please ensure these files are in the same directory as this script.")
    # Exit the script if files are not found, as the app cannot function without them.
    exit()

# Load the trained model and scaler
try:
    model = joblib.load('solar_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# Initialize the Flask application
app = Flask(__name__)

# Define the features that the model was trained on
FEATURES = [
    'distance-to-solar-noon', 'temperature', 'wind-direction', 
    'wind-speed', 'sky-cover', 'visibility', 'humidity', 
    'average-wind-speed-(period)', 'average-pressure-(period)'
]

@app.route('/')
def home():
    """
    Renders the home page of the application.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the user.
    """
    try:
        # Get the input data from the form
        data = request.form.to_dict()
        
        # Convert values to float and create a DataFrame
        # The DataFrame must have the same column order as the training data
        input_data = {
            feature: [float(data.get(feature, 0))] for feature in FEATURES
        }
        df_input = pd.DataFrame(input_data)
        
        # Scale the input data using the pre-trained scaler
        scaled_input = scaler.transform(df_input)

        # Make a prediction using the loaded model
        prediction = model.predict(scaled_input)
        
        # Format the prediction result to 2 decimal places
        predicted_power = round(prediction[0], 2)

        return jsonify({'prediction': predicted_power})

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction. Please check your inputs.'}), 400

if __name__ == '__main__':
    # Run the app locally in debug mode
    app.run(debug=True)
