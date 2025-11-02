import os
import requests
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# ========== FIX: Restrict MLflow tracking path ==========
mlflow.set_tracking_uri("file:./mlruns")  # ✅ logs locally inside repo (no /content issues)

# ========== STEP 1: Ensure dataset exists ==========
csv_path = "data/heart.csv"
data_url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"

if not os.path.exists(csv_path):
    os.makedirs("data", exist_ok=True)
    print("📥 Downloading dataset from:", data_url)
    try:
        response = requests.get(data_url)
        response.raise_for_status()
        with open(csv_path, "wb") as f:
            f.write(response.content)
        print("✅ heart.csv downloaded successfully.")
    except Exception as e:
        print("❌ Failed to download dataset:", e)
        raise SystemExit("Dataset download failed — stopping pipeline.")

# ========== STEP 2: Load and inspect dataset ==========
try:
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset loaded. Shape: {df.shape}")
except Exception as e:
    raise SystemExit(f"❌ Error reading dataset: {e}")

print("\n📋 Columns:", df.columns.tolist())
print(df.head())

# ========== STEP 3: Verify target column ==========
if "target" not in df.columns:
    raise ValueError(f"❌ 'target' column not found. Available columns: {df.columns.tolist()}")

# ========== STEP 4: Split data ==========
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split: {X_train.shape[0]} train / {X_test.shape[0]} test")

# ========== STEP 5: Train model ==========
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# ========== STEP 6: Evaluate model ==========
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "auc": roc_auc_score(y_test, y_prob)
}

print("\n📊 Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ========== STEP 7: Log metrics with MLflow ==========
mlflow.set_experiment("heart_experiment")

with mlflow.start_run():
    mlflow.log_params({
        "model": "RandomForest",
        "max_depth": 5,
        "n_estimators": 100,
        "random_state": 42
    })
    mlflow.log_metrics(metrics)

    # ✅ Log model to local artifacts folder only
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

print("✅ MLflow logging complete.")

# ========== STEP 8: Save trained model ==========
os.makedirs("models", exist_ok=True)
model_path = "models/heart_model.joblib"
joblib.dump(model, model_path)
print(f"✅ Model saved at: {model_path}")
