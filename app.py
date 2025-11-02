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

# Load the model (skip feature_names.joblib if it exists)
model_path = [f for f in model_files if "feature_names" not in f][0]
model = joblib.load(model_path)
st.success(f"✅ Loaded model: {os.path.basename(model_path)}")

# === Define expected columns ===
expected_features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal'
]

# === Collect numeric inputs ===
inputs = {}
inputs['age'] = st.number_input("Age", 20, 100, 50)
inputs['sex'] = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
inputs['cp'] = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
inputs['trestbps'] = st.number_input("Resting Blood Pressure", 80, 200, 120)
inputs['chol'] = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
inputs['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
inputs['restecg'] = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
inputs['thalach'] = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
inputs['exang'] = st.selectbox("Exercise Induced Angina (1=True, 0=False)", [1, 0])
inputs['oldpeak'] = st.number_input("ST Depression", 0.0, 6.0, 1.0)
inputs['slope'] = st.selectbox("Slope (0–2)", [0, 1, 2])
inputs['ca'] = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
inputs['thal'] = st.selectbox("Thalassemia (0=Normal, 1=Fixed, 2=Reversible)", [0, 1, 2])

# === Create DataFrame with correct column order ===
input_df = pd.DataFrame([[inputs[feature] for feature in expected_features]], columns=expected_features)

# === Predict ===
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        st.subheader("Result:")
        if prediction == 1:
            st.error(f"⚠️ The patient is likely to have heart disease. (Probability: {probability:.2f})")
        else:
            st.success(f"✅ The patient is likely healthy. (Probability: {probability:.2f})")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
