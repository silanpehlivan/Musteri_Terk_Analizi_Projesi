"""Optuna-tuned LightGBM + Deep NN OOF stacking (5-fold).

This script:
- Tunes LightGBM hyperparameters with Optuna (n_trials configurable)
- Generates out-of-fold (OOF) predictions for LightGBM and Deep NN
- Trains a meta LogisticRegression on OOF preds
- Evaluates on held-out test set and saves `models/metrics.json`

Note: This can be CPU-intensive. Tune `N_TRIALS` and epochs for speed.
"""
import ospyt
import json
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import optuna

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 30


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


def tune_lgbm(X, y, n_trials=30):
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2)
        }
        # 3-fold CV to speed up
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            clf = lgb.LGBMClassifier(**{k: v for k, v in param.items() if k in lgb.LGBMClassifier().get_params()})
            clf.set_params(n_estimators=200)
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30)])
            y_pred = clf.predict_proba(X_val)[:, 1]
            m = eval_metrics(y_val, y_pred)['f1']
            scores.append(m)
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def main():
    data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    out_dir = 'models'
    os.makedirs(out_dir, exist_ok=True)

    X_df, y_series = load_data(data_path)
    X_train_full, X_test_df, y_train_full, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=RANDOM_STATE, stratify=y_series)

    # Preprocessor
    preproc_path = os.path.join(out_dir, 'preprocessor.joblib')
    if os.path.exists(preproc_path):
        pre = joblib.load(preproc_path)
    else:
        pre = build_preprocessor(X_train_full)
        joblib.dump(pre, preproc_path)

    # Transform full arrays
    X_all = pre.transform(X_train_full)
    X_test = pre.transform(X_test_df)
    y_all = y_train_full.values

    # Tune LightGBM on X_all
    print('Tuning LightGBM with Optuna...')
    best_params = tune_lgbm(X_all, y_all, n_trials=N_TRIALS)
    print('Best params:', best_params)

    # Prepare OOF arrays
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    lgb_oof = np.zeros(len(X_all))
    deep_oof = np.zeros(len(X_all))
    lgb_test_preds = np.zeros(len(X_test))
    deep_test_preds = np.zeros(len(X_test))

    fold = 0
    for train_idx, val_idx in skf.split(X_all, y_all):
        fold += 1
        print(f'Fold {fold}/{N_FOLDS}')
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        # LightGBM with best params
        lgbm = lgb.LGBMClassifier(**best_params, n_estimators=1000)
        lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
        lgb_oof[val_idx] = lgbm.predict_proba(X_val)[:, 1]
        lgb_test_preds += lgbm.predict_proba(X_test)[:, 1] / N_FOLDS

        # Deep NN
        deep = build_deep(X_tr.shape[1])
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        deep.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[es], verbose=0)
        deep_oof[val_idx] = deep.predict(X_val).ravel()
        deep_test_preds += deep.predict(X_test).ravel() / N_FOLDS

    # Meta model
    meta_X = np.vstack([lgb_oof, deep_oof]).T
    meta = LogisticRegression(solver='liblinear')
    meta.fit(meta_X, y_all)

    # Evaluate on test
    meta_test_prob = meta.predict_proba(np.vstack([lgb_test_preds, deep_test_preds]).T)[:, 1]
    lgb_metrics = eval_metrics(y_test, lgb_test_preds)
    deep_metrics = eval_metrics(y_test, deep_test_preds)
    stack_metrics = eval_metrics(y_test, meta_test_prob)

    metrics = {'lightgbm': lgb_metrics, 'deep_nn': deep_metrics, 'stack': stack_metrics}
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Save final models
    with open(os.path.join(out_dir, 'lgbm_churn_model.pkl'), 'wb') as f:
        pickle.dump(lgbm, f)
    deep.save(os.path.join(out_dir, 'deep_churn_model.h5'))
    joblib.dump(meta, os.path.join(out_dir, 'stack_meta.pkl'))

    print('Completed. Metrics:')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
