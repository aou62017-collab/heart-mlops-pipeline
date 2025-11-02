import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
print("\n📋 Columns:", list(df.columns))

df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df = df.drop_duplicates().dropna()

target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"\n📊 Numeric: {numeric_cols}\n🔤 Categorical: {categorical_cols}")

# ========== STEP 3: Build model pipeline ==========
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
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

    # ✅ Add input example to fix schema warnings
    sample_input = X_train.iloc[:1]

    # ✅ Log model to local artifact path
    artifact_subdir = os.path.join(MLFLOW_DIR, "artifacts_model")
    os.makedirs(artifact_subdir, exist_ok=True)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=sample_input
    )

    # ✅ Save model for Streamlit app
    model_path = os.path.join(MODEL_DIR, "heart_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved locally at: {model_path}")

print("\n🚀 Training complete — no permission errors 🎉")
