# ===============================
# FULL WORKING TRAIN.PY (CI SAFE)
# ===============================

import os
import requests
import pandas as pd
import joblib
from datetime import datetime

# Fix for GitHub Actions (no GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

print("🚀 Starting full training pipeline...")

# ===============================
# STEP 1: ENSURE DATASET EXISTS
# ===============================
DATA_DIR = "data"
MODEL_DIR = "models"
BASE_DIR = os.getcwd()

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "heart.csv")
DATA_URL = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"

if not os.path.exists(DATA_PATH):
    print("📥 Downloading dataset ...")
    r = requests.get(DATA_URL)
    r.raise_for_status()
    with open(DATA_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Dataset downloaded")

# ===============================
# STEP 2: LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)

df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

X = df.drop("target", axis=1)
y = df["target"]

print("✅ Data loaded:", X.shape)

# ===============================
# STEP 3: SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 4: BASELINE MODEL
# ===============================
print("⚙️ Training Baseline Model...")
baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])
baseline_model.fit(X_train, y_train)

baseline_preds = baseline_model.predict(X_test)
baseline_probs = baseline_model.predict_proba(X_test)[:, 1]

baseline_metrics = {
    "baseline_accuracy": accuracy_score(y_test, baseline_preds),
    "baseline_precision": precision_score(y_test, baseline_preds),
    "baseline_recall": recall_score(y_test, baseline_preds),
    "baseline_f1": f1_score(y_test, baseline_preds),
    "baseline_auc": roc_auc_score(y_test, baseline_probs)
}

# ===============================
# STEP 5: MLOPS MODEL
# ===============================
print("⚙️ Training Random Forest (MLOps Model)...")

mlops_model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])
mlops_model.fit(X_train, y_train)

preds = mlops_model.predict(X_test)
probs = mlops_model.predict_proba(X_test)[:, 1]

mlops_metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds),
    "recall": recall_score(y_test, preds),
    "f1": f1_score(y_test, preds),
    "auc": roc_auc_score(y_test, probs)
}

# ===============================
# STEP 6: SAVE VERSIONED MODEL
# ===============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"heart_model_v{timestamp}.joblib"
model_path = os.path.join(MODEL_DIR, model_name)

joblib.dump(mlops_model, model_path)

# Save feature names
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.joblib"))

print(f"✅ Model saved: {model_name}")

# ===============================
# STEP 7: CONFUSION MATRIX & ROC
# ===============================
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

print("✅ Confusion matrix and ROC curve saved.")

# ===============================
# STEP 8: UPDATE MODEL REGISTRY
# ===============================
registry_path = "model_registry.csv"

log_row = {
    "timestamp": timestamp,
    "model_path": model_path,
    **baseline_metrics,
    **mlops_metrics
}

log_df = pd.DataFrame([log_row])

if os.path.exists(registry_path):
    log_df.to_csv(registry_path, mode="a", header=False, index=False)
else:
    log_df.to_csv(registry_path, index=False)

print("✅ Model registry updated")

print("🚀 Training pipeline finished successfully!")
