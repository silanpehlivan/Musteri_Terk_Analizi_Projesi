"""Train a simple stacking (blending) ensemble: LightGBM + Deep NN -> LogisticRegression

Approach:
- Load data and preprocessor (or fit if missing)
- Split into train / test
- Further split train into train_base and holdout
- Train LightGBM and a Keras deep model on train_base
- Predict probabilities on holdout, train LogisticRegression meta-model
- Evaluate on test set and save models + updated metrics.json
"""
import os
import json
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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


def build_preprocessor(X_train):
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


def build_deep(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def eval_metrics(y_true, y_pred_prob):
    y_pred = (y_pred_prob >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }


def main():
    data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    out_dir = 'models'
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_data(data_path)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # train_base and holdout
    X_train_base, X_hold, y_train_base, y_hold = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

    # Preprocessor
    preproc_path = os.path.join(out_dir, 'preprocessor.joblib')
    if os.path.exists(preproc_path):
        pre = joblib.load(preproc_path)
    else:
        pre = build_preprocessor(X_train_base)
        joblib.dump(pre, preproc_path)

    Xb = pre.transform(X_train_base)
    Xh = pre.transform(X_hold)
    Xt = pre.transform(X_test)

    # LightGBM on train_base
    print('Training LightGBM on train_base...')
    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42)
    lgbm.fit(Xb, y_train_base)

    # Deep model on train_base
    print('Training Deep NN on train_base...')
    deep = build_deep(Xb.shape[1])
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)]
    deep.fit(Xb, y_train_base, validation_data=(Xh, y_hold), epochs=50, batch_size=64, callbacks=callbacks, verbose=1)

    # Predict on holdout
    lgb_hold_prob = lgbm.predict_proba(Xh)[:, 1]
    deep_hold_prob = deep.predict(Xh).ravel()

    # Meta model
    meta_X = np.vstack([lgb_hold_prob, deep_hold_prob]).T
    meta = LogisticRegression(solver='liblinear')
    meta.fit(meta_X, y_hold)

    # Evaluate on test set
    lgb_test_prob = lgbm.predict_proba(Xt)[:, 1]
    deep_test_prob = deep.predict(Xt).ravel()
    meta_test_prob = meta.predict_proba(np.vstack([lgb_test_prob, deep_test_prob]).T)[:, 1]

    lgb_metrics = eval_metrics(y_test, lgb_test_prob)
    deep_metrics = eval_metrics(y_test, deep_test_prob)
    stack_metrics = eval_metrics(y_test, meta_test_prob)

    metrics = {
        'lightgbm': lgb_metrics,
        'deep_nn': deep_metrics,
        'stack': stack_metrics
    }

    # Save models
    with open(os.path.join(out_dir, 'lgbm_churn_model.pkl'), 'wb') as f:
        pickle.dump(lgbm, f)
    deep.save(os.path.join(out_dir, 'deep_churn_model.h5'))
    joblib.dump(meta, os.path.join(out_dir, 'stack_meta.pkl'))

    # Save metrics
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print('Stack training complete. Metrics:')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
