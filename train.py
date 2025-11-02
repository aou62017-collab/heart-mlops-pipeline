import pandas as pd, yaml, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow, mlflow.sklearn
from model import build_model, save_model

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def preprocess(X):
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype(str).fillna("Unknown")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    return pd.get_dummies(X, drop_first=True)

params = load_params()
df = pd.read_csv(params["data"]["path"])

target_col = "target"
for col in df.columns:
    if col.lower() in ["target", "heartdisease", "output"]:
        target_col = col
        break

y = df[target_col]
X = preprocess(df.drop(columns=[target_col]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["data"]["test_size"], stratify=y, random_state=params["data"]["random_state"]
)

model_params = params["model"]
pipeline = build_model(model_params)

mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
mlflow.set_experiment(params["mlflow"]["experiment_name"])

with mlflow.start_run() as run:
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probs))
    }
    mlflow.log_metrics(metrics)
    save_model(pipeline, f"models/model_{run.info.run_id}.joblib")
    print("✅ Training complete.")
    print("Metrics:", metrics)
