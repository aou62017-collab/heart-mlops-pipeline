import streamlit as st
import pandas as pd
import joblib
import os
import glob

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict the risk of heart disease.")

# === Load latest model ===
model_dir = "models"
model_files = sorted(glob.glob(f"{model_dir}/*.joblib"), key=os.path.getmtime, reverse=True)
if not model_files:
    st.error("❌ No model found. Please train and push a model first.")
    st.stop()
model = joblib.load(model_files[0])
st.success(f"✅ Loaded model: {os.path.basename(model_files[0])}")

# === Collect inputs ===
inputs = {
    "age": st.number_input("Age", 20, 100, 50),
    "sex": 1 if st.selectbox("Sex", ["Male", "Female"]) == "Male" else 0,
    "cp": st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3]),
    "trestbps": st.number_input("Resting Blood Pressure", 80, 200, 120),
    "chol": st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200),
    "fbs": st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1]),
    "restecg": st.selectbox("Resting ECG (0–2)", [0, 1, 2]),
    "thalach": st.number_input("Max Heart Rate Achieved", 60, 220, 150),
    "exang": st.selectbox("Exercise Induced Angina", [0, 1]),
    "oldpeak": st.number_input("ST Depression", 0.0, 6.0, 1.0),
    "slope": st.selectbox("Slope (0–2)", [0, 1, 2]),
    "ca": st.selectbox("No. of Major Vessels (0–3)", [0, 1, 2, 3]),
    "thal": st.selectbox("Thalassemia (0, 1, 2)", [0, 1, 2])
}

# === Create input DataFrame ===
input_df = pd.DataFrame([inputs])

# === Predict ===
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is likely healthy.")
