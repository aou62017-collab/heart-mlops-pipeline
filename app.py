import streamlit as st
import pandas as pd
import joblib
import os
import glob

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict heart disease risk.")

# === Load model and feature names ===
model_dir = "models"
try:
    # Load model
    model_path = os.path.join(model_dir, "heart_model.joblib")
    model = joblib.load(model_path)
    
    # Load feature names
    feature_names_path = os.path.join(model_dir, "feature_names.joblib")
    feature_names = joblib.load(feature_names_path)
    
    st.success(f"✅ Model loaded successfully!")
    st.info(f"Features expected: {', '.join(feature_names)}")
    
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# === Collect inputs in EXACT same order as training features ===
inputs = {}

# Map feature names to input widgets
feature_descriptions = {
    'age': "Age",
    'sex': "Sex (1=Male, 0=Female)", 
    'cp': "Chest Pain Type (0–3)",
    'trestbps': "Resting Blood Pressure",
    'chol': "Serum Cholesterol (mg/dl)",
    'fbs': "Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)",
    'restecg': "Resting ECG (0–2)",
    'thalach': "Max Heart Rate Achieved", 
    'exang': "Exercise Induced Angina (1=True, 0=False)",
    'oldpeak': "ST Depression",
    'slope': "Slope (0–2)",
    'ca': "Number of Major Vessels (0–3)",
    'thal': "Thalassemia (0=Normal, 1=Fixed, 2=Reversible)"
}

# Create inputs in the exact order of feature_names
for feature in feature_names:
    if feature == 'age':
        inputs[feature] = st.number_input(feature_descriptions[feature], 20, 100, 50)
    elif feature == 'sex':
        inputs[feature] = st.selectbox(feature_descriptions[feature], [1, 0])
    elif feature == 'cp':
        inputs[feature] = st.selectbox(feature_descriptions[feature], [0, 1, 2, 3])
    elif feature == 'trestbps':
        inputs[feature] = st.number_input(feature_descriptions[feature], 80, 200, 120)
    elif feature == 'chol':
        inputs[feature] = st.number_input(feature_descriptions[feature], 100, 400, 200)
    elif feature == 'fbs':
        inputs[feature] = st.selectbox(feature_descriptions[feature], [1, 0])
    elif feature == 'restecg':
        inputs[feature] = st.selectbox(feature_descriptions[feature], [0, 1, 2])
    elif feature == 'thalach':
        inputs[feature] = st.number_input(feature_descriptions[feature], 60, 220, 150)
    elif feature == 'exang':
        inputs[feature] = st.selectbox(feature_descriptions[feature], [1, 0])
    elif feature == 'oldpeak':
        inputs[feature] = st.number_input(feature_descriptions[feature], 0.0, 6.0, 1.0)
    elif feature == 'slope':
        inputs[feature] = st.selectbox(feature_descriptions[feature], [0, 1, 2])
    elif feature == 'ca':
        inputs[feature] = st.selectbox(feature_descriptions[feature], [0, 1, 2, 3])
    elif feature == 'thal':
        inputs[feature] = st.selectbox(feature_descriptions[feature], [0, 1, 2])

# === Create DataFrame with EXACT feature order ===
input_df = pd.DataFrame([inputs], columns=feature_names)

# === Predict ===
if st.button("Predict Heart Disease Risk"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        st.subheader("🎯 Prediction Result:")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease (Probability: {probability:.1%})")
            st.write("Recommendation: Please consult a cardiologist for further evaluation.")
        else:
            st.success(f"✅ Low Risk of Heart Disease (Probability: {probability:.1%})")
            st.write("Recommendation: Maintain regular check-ups and a healthy lifestyle.")
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.write("Please ensure all input fields are filled correctly.")
