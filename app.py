import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import requests

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict heart disease risk.")

# === Function to train model if it doesn't exist ===
def train_model_if_missing():
    model_path = "models/heart_model.joblib"
    feature_path = "models/feature_names.joblib"
    
    if os.path.exists(model_path) and os.path.exists(feature_path):
        return True  # Model already exists
    
    st.warning("🔄 Model not found. Training model now... This may take a moment.")
    
    try:
        # Download dataset
        data_url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
        df = pd.read_csv(data_url)
        
        # Clean column names
        df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = df.drop_duplicates().dropna()
        
        # Prepare data
        X = df.drop(columns=['target'])
        y = df['target']
        
        # Create and train model
        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        model.fit(X, y)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model and feature names
        joblib.dump(model, model_path)
        joblib.dump(list(X.columns), feature_path)
        
        st.success("✅ Model trained and saved successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to train model: {e}")
        return False

# === Train model if missing ===
if not train_model_if_missing():
    st.stop()

# === Load model and feature names ===
try:
    model = joblib.load("models/heart_model.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# === User Input Form ===
st.subheader("📝 Patient Information")

inputs = {}
col1, col2 = st.columns(2)

with col1:
    inputs['age'] = st.number_input("Age", min_value=20, max_value=100, value=50)
    inputs['sex'] = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    inputs['cp'] = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                               format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 
                                                     2: "Non-anginal Pain", 3: "Asymptomatic"}[x])
    inputs['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    inputs['chol'] = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
    inputs['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], 
                                format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    inputs['restecg'] = st.selectbox("Resting ECG", options=[0, 1, 2],
                                   format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 
                                                         2: "Left Ventricular Hypertrophy"}[x])
    inputs['thalach'] = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    inputs['exang'] = st.selectbox("Exercise Induced Angina", options=[1, 0],
                                  format_func=lambda x: "Yes" if x == 1 else "No")
    inputs['oldpeak'] = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    inputs['slope'] = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                                  format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    inputs['ca'] = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
    inputs['thal'] = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                                 format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 
                                                       2: "Reversible Defect", 3: "Unknown"}[x])

# === Create DataFrame with correct feature order ===
input_df = pd.DataFrame([inputs], columns=feature_names)

# === Display input data ===
if st.checkbox("Show input data"):
    st.write("📊 Input Data Preview:")
    st.dataframe(input_df)

# === Prediction ===
if st.button("🔍 Predict Heart Disease Risk", type="primary"):
    try:
        with st.spinner("Analyzing patient data..."):
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
        
        st.subheader("🎯 Prediction Result")
        
        if prediction == 1:
            st.error(f"⚠️ **High Risk of Heart Disease**")
            st.write(f"Probability: {probability:.1%}")
            st.write("**Recommendation:** Please consult a cardiologist for further evaluation and treatment.")
        else:
            st.success(f"✅ **Low Risk of Heart Disease**")
            st.write(f"Probability: {probability:.1%}")
            st.write("**Recommendation:** Maintain regular check-ups and a healthy lifestyle.")
            
        # Show confidence level
        if probability > 0.7:
            st.info("Confidence: High")
        elif probability > 0.4:
            st.info("Confidence: Medium")
        else:
            st.info("Confidence: Low")
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.write("Please ensure all input fields are filled correctly.")

# === Footer ===
st.markdown("---")
st.markdown("*Note: This prediction is for educational purposes only. Always consult healthcare professionals for medical advice.*")
