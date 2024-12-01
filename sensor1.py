import random
import numpy as np
import pandas as pd
import joblib
import time

# Load the pre-trained Random Forest model
model = joblib.load('irrigation_model_with_crop.pkl')

# Simulate sensor data for moisture and temperature (excluding crop_cotton)
def simulate_sensor_data():
    """
    Simulate sensor readings for moisture and temperature.
    """
    # Simulating random values for sensor data
    moisture = random.uniform(300, 800)  # Moisture value between 300 and 800
    temperature = random.uniform(10, 40)  # Temperature value between 10°C and 40°C
    # crop_cotton is removed since the model was trained without it
    return [moisture, temperature]

# Main function to simulate the sensor and make predictions
def main():
    print("Starting sensor simulation and prediction...")

    # Simulate continuous sensor readings and predictions
    while True:
        # Simulate sensor readings
        sensor_data = simulate_sensor_data()

        # Convert sensor data to a DataFrame to match the model input format (moisture, temperature)
        sensor_data_df = pd.DataFrame([sensor_data], columns=['moisture', 'temperature'])

        # Ensure the correct data types for prediction
        sensor_data_df['moisture'] = sensor_data_df['moisture'].astype(float)
        sensor_data_df['temperature'] = sensor_data_df['temperature'].astype(float)

        # Predict the pump action based on sensor data
        prediction = model.predict(sensor_data_df.values)  # Use .values to convert DataFrame to NumPy array
        prediction_label = 'ON' if prediction[0] == 1 else 'OFF'

        # Display the sensor data and prediction
        print(f"Sensor Data -> Moisture: {sensor_data[0]:.2f}, Temperature: {sensor_data[1]:.2f}")
        print(f"Predicted Pump Action: {prediction_label}")

        # Simulate a delay between sensor readings (e.g., 2 seconds)
        time.sleep(2)

# Run the main function
if __name__ == "__main__":
    main()

