import os
import requests
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

# MLflow local path (safe for GitHub Actions)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ci_cd_experiment")

print("🚀 Starting model training...")

# ========== STEP 1: Ensure dataset exists ==========
csv_path = "data/heart.csv"
data_url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"

if not os.path.exists(csv_path):
    os.makedirs("data", exist_ok=True)
    print("📥 Downloading dataset...")
    response = requests.get(data_url)
    response.raise_for_status()
    with open(csv_path, "wb") as f:
        f.write(response.content)
    print("✅ Dataset downloaded.")

# ========== STEP 2: Load dataset ==========
df = pd.read_csv(csv_path)

X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== STEP 3: Build Pipeline ==========
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(max_depth=8, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "auc": roc_auc_score(y_test, y_proba)
}

print("📊 Metrics:", metrics)

# ========== STEP 4: Save Model ==========
os.makedirs("models", exist_ok=True)
model_path = f"models/model_{mlflow.active_run().info.run_id}.joblib"
joblib.dump(pipeline, model_path)
print("✅ Model saved:", model_path)

# ========== STEP 5: Safe MLflow Logging ==========
with mlflow.start_run():
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
