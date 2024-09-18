import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Title of the app
st.title('Energy Consumption Prediction')

# Load the data
data = pd.read_csv('Consumption.csv')
st.write("Sample Data:")
st.write(data)

# Prepare features and target variable
features = ['num_residents', 'house_size', 'avg_temperature']
X = data[features]
y = data['energy_consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model parameters
st.write(f"Model Coefficients: {model.coef_}")
st.write(f"Intercept: {model.intercept_}")

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")

# Visualize predictions vs true values
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue')
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
ax.set_xlabel('True Energy Consumption (kWh)')
ax.set_ylabel('Predicted Energy Consumption (kWh)')
ax.set_title('Predictions vs True Values')
st.pyplot(fig)

# User input for prediction
st.header('Input House Details for Prediction')

num_residents_input = st.number_input('Number of Residents', min_value=1, value=3)
house_size_input = st.number_input('House Size (sq ft)', min_value=0, value=1500)
avg_temperature_input = st.number_input('Average Temperature (°C)', min_value=-30, value=22)  # Celsius input

# Prepare input data for prediction
input_data = np.array([[num_residents_input, house_size_input, avg_temperature_input]])
predicted_energy_consumption = model.predict(input_data)

# Display the prediction result
st.write(f'Predicted Energy Consumption: {predicted_energy_consumption[0]:.2f} kWh')
