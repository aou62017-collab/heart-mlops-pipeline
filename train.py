import os
import requests
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Step 1: Ensure dataset exists
csv_path = "data/heart.csv"
if not os.path.exists(csv_path):
    os.makedirs("data", exist_ok=True)
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/heart/heart.csv"
    print("📥 Downloading dataset...")
    response = requests.get(url)
    with open(csv_path, "wb") as f:
        f.write(response.content)
    print("✅ heart.csv downloaded successfully.")

# Step 2: Load dataset
df = pd.read_csv(csv_path)
print(f"✅ Dataset loaded. Shape: {df.shape}")

# Step 3: Split features and target
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split complete:", X_train.shape, X_test.shape)

# Step 4: Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Step 5: Evaluate model
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

# Step 6: Log to MLflow
mlflow.set_experiment("heart_experiment")
with mlflow.start_run():
    mlflow.log_params({"model": "RandomForest", "max_depth": 5, "n_estimators": 100})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")

# Step 7: Save trained model
os.makedirs("models", exist_ok=True)
model_path = "models/heart_model.joblib"
joblib.dump(model, model_path)
print(f"✅ Model saved at: {model_path}")
