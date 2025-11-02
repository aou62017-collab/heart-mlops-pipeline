import streamlit as st
import pandas as pd
import joblib
import os
import glob

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict the risk of heart disease.")

# === Step 1: Auto-detect the most recent model file ===
model_dir = "models"
model_files = sorted(glob.glob(f"{model_dir}/*.joblib"), key=os.path.getmtime, reverse=True)

if not model_files:
    st.error("❌ No model file found in models/. Please train and push your model first.")
    st.stop()
else:
    model_path = model_files[0]
    model = joblib.load(model_path)
    st.success(f"✅ Loaded model: {os.path.basename(model_path)}")

# === Step 2: Input form ===
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0=normal, 1=fixed defect, 2=reversible defect)", [0, 1, 2])

# === Step 3: Prepare input data ===
input_data = pd.DataFrame([[
    age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]], columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
             "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

# === Step 4: Prediction ===
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is likely healthy.")
