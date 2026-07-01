import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")
scaler=joblib.load("scaler.pkl")
# Title
st.title("Diabetes Prediction")

# User Inputs
preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

# Prediction
if st.button("Predict"):

    features = np.array([[preg, glucose, bp, skin,
                          insulin, bmi, dpf, age]])
    features=scaler.transform(features)
    prediction = model.predict(features)
    probability=model.predict_proba(features)

    if prediction[0] == 1:
        st.error("Person is likely to have Diabetes")
    else:
        st.success("Person is not likely to have Diabetes")
    st.write("Prediction Probability:")
    st.write(probability)