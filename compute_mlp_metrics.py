import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODEL_DIR = 'models'
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MLP_PATH = os.path.join(MODEL_DIR, 'mlp_churn_model.h5')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Data file not found: {DATA_PATH}")

# Load data and reproduce notebook preprocessing
df = pd.read_csv(DATA_PATH)
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# map churn
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Binary mapping
binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    if col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})

# One-hot encode categorical columns used in notebook
categorical_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaymentMethod'
]
for c in categorical_cols:
    if c in df.columns:
        df[c] = df[c].astype(str)

# perform get_dummies like the notebook
df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True)

# split
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load scaler from models/scaler.pkl (notebook saved it)
if not os.path.exists(SCALER_PATH):
    raise SystemExit(f"Scaler not found: {SCALER_PATH}")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numerical_cols:
    if col in X_test.columns:
        X_test[col] = scaler.transform(X_test[[col]]) if hasattr(scaler, 'transform') and False else scaler.transform(X_test[numerical_cols])
        break
# The above line is a safe guard; do proper scaling below
if all(c in X_test.columns for c in numerical_cols):
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Load MLP model
if not os.path.exists(MLP_PATH):
    raise SystemExit(f"MLP model not found: {MLP_PATH}")
mlp = load_model(MLP_PATH)

# Predict
y_prob = mlp.predict(X_test).ravel()
THRESHOLD = 0.60
y_pred = (y_prob >= THRESHOLD).astype(int)

metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
    'f1': float(f1_score(y_test, y_pred, zero_division=0))
}

# Load existing metrics.json if exists
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        existing = json.load(f)
else:
    existing = {}

existing['jupyter_mlp'] = metrics

with open(METRICS_PATH, 'w', encoding='utf-8') as f:
    json.dump(existing, f, indent=2)

print('Wrote jupyter_mlp metrics to', METRICS_PATH)
print(json.dumps(metrics, indent=2))
