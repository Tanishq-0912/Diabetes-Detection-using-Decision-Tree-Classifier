
import streamlit as st
import numpy as np
import pickle
import joblib

# Load the trained model
joblib.dump(dt_model, "dt_model.pkl")

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient details below to predict the likelihood of diabetes.")

# Input fields (based on Pima Indians Diabetes dataset features)
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.number_input("Age", min_value=10, max_value=120, value=33)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    prediction = dt_model.predict(input_data)[0]
    class_names = ['Diabitic', 'Non Diabitic']
    st.success(f"🌼 Predicted class: {class_names[prediction]}")
