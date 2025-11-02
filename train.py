import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import requests

# ========= STEP 0: Set Safe Paths ==========
BASE_DIR = os.getcwd()
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns_local")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MLFLOW_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Force MLflow to log everything in ./mlruns_local
mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
mlflow.set_experiment("heart_experiment")

# ========== STEP 1: Ensure dataset exists ==========
csv_path = "data/heart.csv"
data_url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"

if not os.path.exists(csv_path):
    os.makedirs("data", exist_ok=True)
    print("📥 Downloading dataset from:", data_url)
    response = requests.get(data_url)
    response.raise_for_status()
    with open(csv_path, "wb") as f:
        f.write(response.content)
    print("✅ heart.csv downloaded successfully.")

# ========== STEP 2: Load and preprocess ==========
df = pd.read_csv(csv_path)
print("✅ Dataset loaded. Shape:", df.shape)
print("\n📋 Original Columns:", list(df.columns))

# Clean column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df = df.drop_duplicates().dropna()

print("🧹 Cleaned Columns:", list(df.columns))
print("\n🔍 Data types:")
print(df.dtypes)

target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col]

# ✅ FIX: All columns in heart dataset are numeric, no categorical columns
print(f"\n📊 All features are numeric: {list(X.columns)}")

# ========== STEP 3: Build model pipeline ==========
# ✅ FIX: Use simple StandardScaler since all features are numeric
model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42))
])

# ========== STEP 4: Train/test split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split: {len(X_train)} train / {len(X_test)} test")

# ========== STEP 5: Train, Evaluate, Log ==========
with mlflow.start_run():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs),
    }

    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        mlflow.log_metric(k, v)

    # ✅ Log input example
    sample_input = X_train.iloc[:1]

    # ✅ Log model to MLflow
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=sample_input
    )

    # ✅ Save model for Streamlit app
    model_path = os.path.join(MODEL_DIR, "heart_model.joblib")
    joblib.dump(model, model_path)
    
    # ✅ ALSO save the feature names for reference
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    joblib.dump(list(X.columns), feature_names_path)
    
    print(f"\n✅ Model saved locally at: {model_path}")
    print(f"✅ Feature names saved at: {feature_names_path}")

print("\n🚀 Training complete — model is now compatible with Streamlit app 🎉")
