import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

print("🚀 Starting full training pipeline...")

# --------------------------
# Paths
# --------------------------
DATA_PATH = "data/heart.csv"
MODEL_DIR = "models"
REGISTRY_PATH = "model_registry.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv(DATA_PATH)

df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

X = df.drop("target", axis=1)
y = df["target"]

# --------------------------
# Split data
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# Baseline Model (Logistic Regression)
# --------------------------
print("\n⚙️ Training Baseline Model (Logistic Regression)...")

baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

baseline_model.fit(X_train, y_train)

baseline_preds = baseline_model.predict(X_test)
baseline_probs = baseline_model.predict_proba(X_test)[:, 1]

baseline_accuracy = accuracy_score(y_test, baseline_preds)
baseline_precision = precision_score(y_test, baseline_preds)
baseline_recall = recall_score(y_test, baseline_preds)
baseline_f1 = f1_score(y_test, baseline_preds)
baseline_auc = roc_auc_score(y_test, baseline_probs)

print("\n📊 Baseline Model Metrics:")
print(f"baseline_accuracy: {baseline_accuracy:.4f}")
print(f"baseline_precision: {baseline_precision:.4f}")
print(f"baseline_recall: {baseline_recall:.4f}")
print(f"baseline_f1: {baseline_f1:.4f}")
print(f"baseline_auc: {baseline_auc:.4f}")

# --------------------------
# MLOps Model (Random Forest)
# --------------------------
print("\n⚙️ Training Random Forest (MLOps Pipeline)...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc_score = roc_auc_score(y_test, probs)

print("\n📊 MLOps Model Metrics:")
print(f"accuracy: {accuracy:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall: {recall:.4f}")
print(f"f1: {f1:.4f}")
print(f"auc: {auc_score:.4f}")

# --------------------------
# Save MLOps model
# --------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"heart_model_v{timestamp}.joblib"
model_path = os.path.join(MODEL_DIR, model_name)

joblib.dump(model, model_path)

# --------------------------
# Update model registry
# --------------------------
log = pd.DataFrame([{
    "timestamp": timestamp,
    "model_name": model_name,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "auc": auc_score
}])

if os.path.exists(REGISTRY_PATH):
    log.to_csv(REGISTRY_PATH, mode="a", header=False, index=False)
else:
    log.to_csv(REGISTRY_PATH, index=False)

print(f"\n✅ Model saved: {model_name}")

# -----------------------------
# Generate and Save Evaluation Plots
# -----------------------------

print("\n📊 Generating evaluation plots...")

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

print("✅ Confusion matrix and ROC curve saved.")

# --------------------------
# Finish
# --------------------------
print("\n✅ Full training pipeline completed successfully.")
