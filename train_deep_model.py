#!/usr/bin/env python3
"""Train a deeper MLP for Telco churn (CPU-friendly).

Usage: python train_deep_model.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Creates:
- models/deep_churn_model.h5
- models/preprocessor.joblib
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_and_preprocess(path):
    df = pd.read_csv(path)
    # Drop customer id if exists
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # TotalCharges may be empty strings
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        raise ValueError('Dataset does not contain Churn column')

    # Split features
    X = df.drop(columns=['Churn'])
    y = df['Churn'].astype(int)

    # Identify numeric and categorical
    numeric_feats = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_feats = X.select_dtypes(include=['object']).columns.tolist()

    # Build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_feats),
            ('cat', categorical_transformer, categorical_feats)
        ],
        remainder='drop'
    )

    return X, y, preprocessor


def build_model(input_dim):
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

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main(args):
    data_path = args.data
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print('Loading and preprocessing data...')
    X, y, preprocessor = load_and_preprocess(data_path)

    # Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit preprocessor on training
    preprocessor.fit(X_train_full)
    X_train = preprocessor.transform(X_train_full)
    X_test_t = preprocessor.transform(X_test)

    # Handle class imbalance with SMOTE on training set
    print('Applying SMOTE to balance classes on training set...')
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train_full)

    print('Resampled training shape:', X_res.shape, 'Positives:', int(y_res.sum()), 'Total:', len(y_res))

    # Build model
    print('Building model...')
    model = build_model(X_res.shape[1])

    # Callbacks
    checkpoint_path = os.path.join(out_dir, 'deep_churn_model.h5')
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
    ]

    print('Training...')
    history = model.fit(
        X_res, y_res,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print('Evaluating on test set...')
    y_pred_prob = model.predict(X_test_t).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print('Classification Report:')
    print(classification_report(y_test, y_pred, digits=4))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    # Save preprocessing objects
    joblib.dump(preprocessor, os.path.join(out_dir, 'preprocessor.joblib'))
    print('Saved model and preprocessor to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    parser.add_argument('--out-dir', type=str, default='models')
    args = parser.parse_args()
    main(args)
