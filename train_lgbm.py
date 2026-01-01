"""Train a LightGBM model on the Telco churn dataset and save metrics for both models.

Produces:
- models/lgbm_churn_model.pkl
- models/metrics.json

This script will also evaluate the previously trained Keras model (if exists) using the same preprocessing
so we can compare metrics and write them to `models/metrics.json` for the frontend.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb

from tensorflow.keras.models import load_model


def load_data(path):
    df = pd.read_csv(path)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn'])
    y = df['Churn'].astype(int)
    return X, y


def build_or_load_preprocessor(X_train, preproc_path=None):
    if preproc_path and os.path.exists(preproc_path):
        pre = joblib.load(preproc_path)
        return pre

    numeric_feats = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_feats = X_train.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    pre = ColumnTransformer([
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ], remainder='drop')

    pre.fit(X_train)
    return pre


def eval_classifier(clf, X_test, y_test):
    y_prob = None
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        # assume keras model
        y_prob = clf.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    }


def main():
    data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    out_dir = 'models'
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_data(data_path)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preproc_path = os.path.join(out_dir, 'preprocessor.joblib')
    pre = build_or_load_preprocessor(X_train_full, preproc_path if os.path.exists(preproc_path) else None)

    X_train = pre.transform(X_train_full)
    X_test_t = pre.transform(X_test)

    # LightGBM dataset
    # Estimate scale_pos_weight
    neg = (y_train_full == 0).sum()
    pos = (y_train_full == 1).sum()
    scale_pos_weight = neg / max(1, pos)

    print('Training LightGBM (sklearn API) with early stopping...')
    gbm = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        class_weight=None
    )

    gbm.fit(
        X_train,
        y_train_full,
        eval_set=[(X_train, y_train_full)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )

    # Save LightGBM model
    import pickle
    with open(os.path.join(out_dir, 'lgbm_churn_model.pkl'), 'wb') as f:
        pickle.dump(gbm, f)

    # Evaluate LightGBM on test
    lgb_metrics = eval_classifier(gbm, X_test_t, y_test)

    # Evaluate Keras deep model if available
    deep_metrics = None
    deep_model_path = os.path.join(out_dir, 'deep_churn_model.h5')
    if os.path.exists(deep_model_path):
        try:
            deep = load_model(deep_model_path)
            deep_metrics = eval_classifier(deep, X_test_t, y_test)
        except Exception as e:
            print('Could not load deep model for eval:', e)

    metrics = {
        'lightgbm': lgb_metrics,
        'deep_nn': deep_metrics
    }

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save preprocessor if not exists
    joblib.dump(pre, preproc_path)

    print('Saved LightGBM model and metrics to', out_dir)


if __name__ == '__main__':
    main()
