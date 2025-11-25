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

# Clean column names to match what Streamlit will send
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = df.drop_duplicates().dropna()

print("🧹 Cleaned Columns:", list(df.columns))

target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\n📊 Features for training: {list(X.columns)}")

# ========== STEP 3: Build SIMPLE model pipeline ==========
# No ColumnTransformer, just simple scaling
model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# ========== STEP 4: Train/test split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split: {len(X_train)} train / {len(X_test)} test")

# ========== STEP 5: Train and Save ==========
with mlflow.start_run():
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
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

    # ✅ Save model for Streamlit app
    model_path = os.path.join(MODEL_DIR, "heart_model.joblib")
    joblib.dump(model, model_path)
    
    # ✅ Save feature names
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    joblib.dump(list(X.columns), feature_names_path)
    # ========== STEP 6: Save training logs for monitoring ==========
log_path = os.path.join(BASE_DIR, "model_registry.csv")

log_data = pd.DataFrame([{
    "timestamp": pd.Timestamp.now(),
    "model_path": model_path,
    "accuracy": metrics["accuracy"],
    "precision": metrics["precision"],
    "recall": metrics["recall"],
    "f1_score": metrics["f1"],
    "auc": metrics["auc"]
}])

if not os.path.exists(log_path):
    log_data.to_csv(log_path, index=False)
else:
    log_data.to_csv(log_path, mode="a", header=False, index=False)

print("✅ Model registry updated")

    
    print(f"\n✅ Model saved at: {model_path}")
    print(f"✅ Feature names: {list(X.columns)}")

print("\n🚀 Training complete! Now run: streamlit run app.py")
