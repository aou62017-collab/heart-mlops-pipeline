import os
import requests
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("♻️ Starting automated retraining pipeline...")

# ========== PATHS ==========
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "heart.csv")
MODEL_DIR = "models"
REGISTRY_PATH = "model_registry.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ========== DOWNLOAD DATA IF NOT EXISTS ==========
data_url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"

if not os.path.exists(DATA_PATH):
    print("📥 Downloading dataset...")
    response = requests.get(data_url)
    response.raise_for_status()
    with open(DATA_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Dataset downloaded.")

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH)

df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

X = df.drop("target", axis=1)
y = df["target"]

# ========== SIMULATE DRIFT ==========
drift_detected = True

if drift_detected:
    print("⚠️ Drift detected — Retraining triggered")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # ========== SAVE VERSIONED MODEL ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"heart_model_v{timestamp}.joblib"
    model_path = os.path.join(MODEL_DIR, model_name)

    joblib.dump(model, model_path)

    # ========== UPDATE MODEL REGISTRY ==========
    log = pd.DataFrame([{
        "timestamp": timestamp,
        "model_path": model_path,
        "accuracy": acc
    }])

    if os.path.exists(REGISTRY_PATH):
        log.to_csv(REGISTRY_PATH, mode="a", header=False, index=False)
    else:
        log.to_csv(REGISTRY_PATH, index=False)

    print(f"✅ New model saved: {model_name}")
    print(f"📊 Model accuracy: {acc}")

print("✅ Retraining completed.")
