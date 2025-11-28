import os
import requests
import pandas as pd
import joblib
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

# -----------------------------
# STEP 1: Auto download dataset (CI Fix)
# -----------------------------
print("🚀 Starting full training pipeline...")

DATA_PATH = "data/heart.csv"
DATA_URL = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"

if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    print("📥 Downloading dataset...")
    r = requests.get(DATA_URL)
    r.raise_for_status()
    with open(DATA_PATH, "wb") as f
