import streamlit as st
import pickle
import numpy as np

# Load your trained model
with open("diabetes_tracking_app.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Diabetes Prediction App")

# User inputs
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=50.0, step=0.1)
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=200.0, step=1.0)
insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, step=1.0)

# Display inputs
st.write(f"**DPF:** {dpf}")
st.write(f"**BMI:** {bmi}")
st.write(f"**Glucose:** {glucose}")
st.write(f"**Insulin:** {insulin}")

# Prediction
if st.button("Predict"):
    features = np.array([[dpf, bmi, glucose, insulin]])
    pred = model.predict(features)[0]
    if pred == 1:
        st.error("🛑 Prediction: Diabetes POSITIVE")
    else:
        st.success("✅ Prediction: Diabetes NEGATIVE")
