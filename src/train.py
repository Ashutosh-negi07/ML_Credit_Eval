"""
Train the credit risk pipeline and save it to models/.

Usage:
    python src/train.py
"""

import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "german_credit_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_pipeline.pkl")

# --- Load & clean data ---
df = pd.read_csv(DATA_PATH)
df["Saving accounts"] = df["Saving accounts"].fillna("Unknown")
df["Checking account"] = df["Checking account"].fillna("Unknown")
df["Risk"] = df["Risk"].str.strip()
df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})

# --- Feature definitions ---
numeric_features = ["Age", "Credit amount", "Duration", "Job"]
categorical_features = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]

X = df.drop("Risk", axis=1)
y = df["Risk"]

# --- Build pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)

# --- Quick evaluation on a hold-out split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Hold-out evaluation:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Good", "Bad"]))

# --- Retrain on full data and save ---
pipeline.fit(X, y)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"Pipeline saved → {MODEL_PATH}")
