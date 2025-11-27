import os
import requests
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


# ========= STEP 0: Set Safe Paths ==========
BASE_DIR = os.getcwd()
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns_local")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MLFLOW_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
mlflow.set_experiment("heart_experiment")

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


# ========== STEP 2: Load and preprocess ==========
df = pd.read_csv(csv_path)

df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = df.drop_duplicates().dropna()

target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"✅ Dataset loaded. Shape: {df.shape}")
print(f"📊 Features: {list(X.columns)}")


# ========== STEP 3: Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ========== STEP 4: BASELINE MODEL ==========
print("\n🔹 Training Baseline Model (Logistic Regression)...")

baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

baseline_model.fit(X_train, y_train)
baseline_preds = baseline_model.predict(X_test)
baseline_probs = baseline_model.predict_proba(X_test)[:, 1]

baseline_metrics = {
    "baseline_accuracy": accuracy_score(y_test, baseline_preds),
    "baseline_precision": precision_score(y_test, baseline_preds),
    "baseline_recall": recall_score(y_test, baseline_preds),
    "baseline_f1": f1_score(y_test, baseline_preds),
    "baseline_auc": roc_auc_score(y_test, baseline_probs),
}

print("\n📊 Baseline Model Metrics:")
for k, v in baseline_metrics.items():
    print(f"{k}: {v:.4f}")


# ========== STEP 5: Main MLOps Model (Random Forest) ==========
print("\n⚙️ Training Random Forest (MLOps Pipeline)...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

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

    print("\n📊 MLOps Model Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        mlflow.log_metric(k, v)


# ========== STEP 6: Save Production Model ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"heart_model_v{timestamp}.joblib"
model_path = os.path.join(MODEL_DIR, model_name)

joblib.dump(model, model_path)

joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.joblib"))

print(f"\n✅ Production model saved: {model_name}")


# ========== STEP 7: Update Model Registry ==========
registry_path = os.path.join(BASE_DIR, "model_registry.csv")

log_data = pd.DataFrame([{
    "timestamp": timestamp,
    "model_path": model_path,
    "accuracy": metrics["accuracy"],
    "precision": metrics["precision"],
    "recall": metrics["recall"],
    "f1_score": metrics["f1"],
    "auc": metrics["auc"],

    # Baseline Metrics
    "baseline_accuracy": baseline_metrics["baseline_accuracy"],
    "baseline_precision": baseline_metrics["baseline_precision"],
    "baseline_recall": baseline_metrics["baseline_recall"],
    "baseline_f1": baseline_metrics["baseline_f1"],
    "baseline_auc": baseline_metrics["baseline_auc"],
}])

if not os.path.exists(registry_path):
    log_data.to_csv(registry_path, index=False)
else:
    log_data.to_csv(registry_path, mode="a", header=False, index=False)

print("📦 Model registry updated successfully.")

print("\n🚀 Training complete! Now run: streamlit run app.py")
