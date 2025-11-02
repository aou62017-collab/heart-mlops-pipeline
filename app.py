import streamlit as st
import pandas as pd
import joblib
import os
import glob

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict heart disease risk.")

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
    "chest_pain_type": st.selectbox("Chest Pain Type", 
        ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]),
    "resting_blood_pressure": st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120),
    "cholestoral": st.number_input("Serum Cholestoral (mg/dl)", 100, 400, 200),
    "fasting_blood_sugar": st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1]),
    "rest_ecg": st.selectbox("Resting ECG (0–2)", [0, 1, 2]),
    "Max_heart_rate": st.number_input("Max Heart Rate Achieved", 60, 220, 150),
    "exercise_induced_angina": st.selectbox("Exercise Induced Angina", [0, 1]),
    "oldpeak": st.number_input("ST Depression", 0.0, 6.0, 1.0),
    "slope": st.selectbox("Slope (0–2)", [0, 1, 2]),
    "ca": st.selectbox("No. of Major Vessels (0–3)", [0, 1, 2, 3]),
    "thal": st.selectbox("Thalassemia (0, 1, 2)", [0, 1, 2])
}

# === Create input DataFrame ===
input_df = pd.DataFrame([inputs])

# === Predict ===
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.subheader("Result:")
        if prediction == 1:
            st.error("⚠️ The patient is likely to have heart disease.")
        else:
            st.success("✅ The patient is likely healthy.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
