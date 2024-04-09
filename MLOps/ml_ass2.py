import streamlit as st
import numpy as np
import pickle

# Load the trained model from the pickle file
model_pkl_file = "C:\\Users\\keert\\Downloads\\trained_model.sav"
with open(model_pkl_file, 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit app title and description
st.write("# Wine Quality Predictor")
st.write("This app predicts the quality of wine based on various attributes.")

# Input fields for user input
st.sidebar.header("Enter Wine Attributes")

fixed_acidity = st.sidebar.number_input("Fixed Acidity", min_value=0.0)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", min_value=0.0)
citric_acid = st.sidebar.number_input("Citric Acid", min_value=0.0)
residual_sugar = st.sidebar.number_input("Residual Sugar", min_value=0.0)
chlorides = st.sidebar.number_input("Chlorides", min_value=0.0)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", min_value=0.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", min_value=0.0)
density = st.sidebar.number_input("Density", min_value=0.0)
pH = st.sidebar.number_input("pH", min_value=0.0)
sulphates = st.sidebar.number_input("Sulphates", min_value=0.0)

# Predict button
if st.sidebar.button("Predict"):
    # Convert user input into array for prediction
    input_data = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                           chlorides, free_sulfur_dioxide, total_sulfur_dioxide, 
                           density, pH, sulphates]).reshape(1, -1)
   
    # Make prediction
    prediction = loaded_model.predict(input_data)

    # Display prediction result
    class_labels = ['1st quality', '2nd quality', '3rd quality', '4th quality', '5th quality', '6th quality', '7th quality', '8th quality', '9th quality', '10th quality']
    st.success(f"The predicted wine quality is: {class_labels[prediction[0]-1]}")
