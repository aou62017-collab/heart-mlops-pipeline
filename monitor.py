import os
import pandas as pd

# Load model registry log
log_file = "model_registry.csv"

if not os.path.exists(log_file):
    print("❌ No model registry found.")
    exit()

df = pd.read_csv(log_file)

latest = df.iloc[-1]
accuracy = latest["accuracy"]

print(f"📊 Latest model accuracy: {accuracy}")

# Drift threshold
DRIFT_THRESHOLD = 0.75

if accuracy < DRIFT_THRESHOLD:
    print("⚠️ Drift detected! Model performance dropped.")
    exit(1)  # Fail pipeline
else:
    print("✅ No drift detected. Model is healthy.")

