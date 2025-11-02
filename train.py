import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ====== STEP 1: Load dataset (auto-download if missing) ======
csv_path = "data/heart.csv"
data_url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"

if not os.path.exists(csv_path):
    os.makedirs("data", exist_ok=True)
    print(f"📥 Downloading dataset from: {data_url}")
    import requests
    try:
        response = requests.get(data_url)
        response.raise_for_status()
        with open(csv_path, "wb") as f:
            f.write(response.content)
        print("✅ heart.csv downloaded successfully.")
    except Exception as e:
        print("❌ Failed to download dataset:", e)
        raise SystemExit("Dataset download failed — stopping pipeline.")

# ====== STEP 2: Prepare Data ======
df = pd.read_csv(csv_path)
print(f"✅ Dataset loaded. Shape: {df.shape}")
print("\n📋 Columns:", list(df.columns))
print(df.head())

# Rename consistent with app
df.rename(columns={
    'trestbps': 'trestbps',
    'chol': 'chol',
    'thalach': 'thalach',
    'target': 'target'
}, inplace=True)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split: {len(X_train)} train / {len(X_test)} test")

# ====== STEP 3: Define ML pipeline ======
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# ====== STEP 4: MLflow tracking ======
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("HeartDiseasePrediction")

with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)
        print(f"{k}: {v:.4f}")

    mlflow.sklearn.log_model(pipeline, "model")

# ====== STEP 5: Save the pipeline model ======
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "heart_model.joblib")
joblib.dump(pipeline, model_path)
print(f"✅ Model saved at {model_path}")
