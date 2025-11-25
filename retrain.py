import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("♻️ Starting automated retraining...")

# Paths
DATA_PATH = "data/heart.csv"
MODEL_DIR = "models"
REGISTRY_PATH = "model_registry.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

X = df.drop("target", axis=1)
y = df["target"]

# Simulate drift trigger (always true for this experiment)
drift_detected = True

if drift_detected:
    print("⚠️ Drift detected — Retraining triggered")

    # Train new model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Save new model version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"heart_model_v{timestamp}.joblib"
    model_path = os.path.join(MODEL_DIR, model_name)

    joblib.dump(model, model_path)

    # Update registry
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
    print(f"📊 New model accuracy: {acc}")

print("✅ Retraining completed.")
