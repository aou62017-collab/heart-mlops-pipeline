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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


print("🚀 Starting model training...")

# ========= PATHS =========
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns_local")
REGISTRY_PATH = os.path.join(BASE_DIR, "model_registry.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
mlflow.set_experiment("heart_experiment")

# ========= STEP 1: Ensure dataset exists =========
csv_path = os.path.join(DATA_DIR, "heart.csv")
data_url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"

if not os.path.exists(csv_path):
    print("📥 Downloading dataset...")
    response = requests.get(data_url)
    response.raise_for_status()
    with open(csv_path, "wb") as f:
        f.write(response.content)
    print("✅ Dataset downloaded.")

# ========= STEP 2: Load and preprocess =========
df = pd.read_csv(csv_path)

df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = df.drop_duplicates().dropna()

X = df.drop("target", axis=1)
y = df["target"]

print("✅ Data loaded. Samples:", len(df))

# ========= STEP 3: Train/test split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========= STEP 4: Build pipeline =========
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# ========= STEP 5: Train + log =========
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

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    print("📊 Metrics:", metrics)

# ========= STEP 6: Save PRIMARY model =========
joblib.dump(model, os.path.join(MODEL_DIR, "heart_model.joblib"))

# Save features
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.joblib"))

# ========= STEP 7: Save VERSIONED model =========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
versioned_name = f"heart_model_v{timestamp}.joblib"
versioned_path = os.path.join(MODEL_DIR, versioned_name)

joblib.dump(model, versioned_path)

print(f"✅ Versioned model saved: {versioned_name}")

# ========= STEP 8: Update model registry =========
log = pd.DataFrame([{
    "timestamp": timestamp,
    "model_path": versioned_path,
    "accuracy": metrics["accuracy"],
    "precision": metrics["precision"],
    "recall": metrics["recall"],
    "f1_score": metrics["f1"],
    "auc": metrics["auc"]
}])

if os.path.exists(REGISTRY_PATH):
    log.to_csv(REGISTRY_PATH, mode="a", header=False, index=False)
else:
    log.to_csv(REGISTRY_PATH, index=False)

print("✅ model_registry.csv updated")

print("\n🚀 Training completed successfully.")
