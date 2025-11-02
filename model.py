import joblib, os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def build_model(params):
    mtype = params.get("type", "random_forest")
    if mtype == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 8)),
            random_state=int(params.get("random_state", 42))
        )
    else:
        clf = LogisticRegression(max_iter=1000, random_state=int(params.get("random_state", 42)))
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

def save_model(pipeline, path="models/model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
