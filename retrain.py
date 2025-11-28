import os
import requests
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
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
    print("📥 Downloading dataset from:", data_url)
    response = requests.get(data_url)
    response.raise_for_status()
    with open(csv_path, "wb") as f:
        f.write(response.content)
    print("✅ heart.csv downloaded successfully.")


# ========== STEP 2: Load and preprocess ==========
df = pd.read_csv(csv_path)
print("✅ Dataset loaded. Shape:", df.shape)

df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = df.drop_duplicates().dropna()
print("🧹 Cleaned Columns:", list(df.columns))

target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\n📊 Features for training: {list(X.columns)}")


# ========== STEP 3: Build Models ==========
baseline_model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

mlops_model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])


# ========== STEP 4: Train/test split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Data split: {len(X_train)} train / {len(X_test)} test")


# ========== STEP 5 + 6 + 7: Train, Evaluate, Save, Analyze ==========
with mlflow.start_run():

    # --------- Baseline Model ---------
    print("\n⚙️ Training Baseline Model (Logistic Regression)...")
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
        mlflow.log_metric(k, v)

    # --------- MLOps Model ---------
    print("\n⚙️ Training Random Forest (MLOps Pipeline)...")
    mlops_model.fit(X_train, y_train)

    preds = mlops_model.predict(X_test)
    probs = mlops_model.predict_proba(X_test)[:, 1]

    mlops_metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs),
    }

    print("\n📊 MLOps Model Metrics:")
    for k, v in mlops_metrics.items():
        print(f"{k}: {v:.4f}")
        mlflow.log_metric(k, v)

    # ========== STEP 7A: Confusion Matrix ==========
    cm = confusion_matrix(y_test, preds)
    print("\n🧩 Confusion Matrix:\n", cm)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ========== STEP 7B: ROC Curve ==========
    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve - Random Forest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("roc_curve.png")
    plt.close()

    print("✅ Confusion matrix and ROC curve saved.")

    # ========== STEP 8: Save model ==========
    model_path = os.path.join(MODEL_DIR, "heart_model.joblib")
    joblib.dump(mlops_model, model_path)

    feature_names_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    joblib.dump(list(X.columns), feature_names_path)

    print(f"\n✅ Model saved at: {model_path}")
    print("✅ Feature names saved")


print("\n🚀 Training complete! Now run: streamlit run app.py")
