# app.py

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import os
import json
import sys
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- Load models and preprocessor ---
MODEL_DIR = 'models'
try:
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
except Exception as e:
    print(f'Warning: preprocessor not found: {e}')
    preprocessor = None

try:
    lgbm_model = None
    lgbm_path = os.path.join(MODEL_DIR, 'lgbm_churn_model.pkl')
    if os.path.exists(lgbm_path):
        with open(lgbm_path, 'rb') as f:
            lgbm_model = pickle.load(f)

    deep_model = None
    deep_path = os.path.join(MODEL_DIR, 'deep_churn_model.h5')
    if os.path.exists(deep_path):
        deep_model = load_model(deep_path)

    stack_meta = None
    meta_path = os.path.join(MODEL_DIR, 'stack_meta.pkl')
    if os.path.exists(meta_path):
        # meta was saved with joblib in training scripts
        stack_meta = joblib.load(meta_path)

    print('Models loaded:', 'lgbm=' + str(lgbm_model is not None), 'deep=' + str(deep_model is not None), 'stack=' + str(stack_meta is not None))
except Exception as e:
    print(f'Error loading models: {e}')
    sys.exit(1)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        with open(os.path.join(MODEL_DIR, 'metrics.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'Error loading metrics: {str(e)}'}), 500


def ensure_columns(df, expected_cols, numeric_cols=None):
    """Ensure DataFrame has expected columns; fill defaults for missing ones."""
    numeric_cols = numeric_cols or []
    for c in expected_cols:
        if c not in df.columns:
            if c in numeric_cols:
                df[c] = 0
            else:
                df[c] = 'No'
    return df


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        raw = pd.DataFrame([data])

        # Ensure numeric types for common numeric fields
        for c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
            if c in raw.columns:
                raw[c] = pd.to_numeric(raw[c], errors='coerce').fillna(0)

        # If preprocessor is available, align input columns
        if preprocessor is not None:
            # Try to extract expected column names from transformer spec
            expected = None
            numeric_cols = []
            try:
                # ColumnTransformer stores the originally passed columns
                expected = []
                for name, trans, cols in preprocessor.transformers_:
                    if cols is None:
                        continue
                    # cols may be a slice or list
                    if isinstance(cols, (list, tuple)):
                        expected.extend(list(cols))
                    else:
                        # If it's an array-like
                        try:
                            expected.extend(list(cols))
                        except Exception:
                            pass
                    if name == 'num':
                        try:
                            numeric_cols.extend(list(cols))
                        except Exception:
                            pass
            except Exception:
                expected = None

            if expected:
                raw = ensure_columns(raw, expected, numeric_cols=numeric_cols)
                X = preprocessor.transform(raw)
            else:
                # Fallback: try to transform original raw (may break if columns missing)
                X = preprocessor.transform(raw)
        else:
            # No preprocessor: attempt to build simple feature vector using available keys
            # This is a degraded fallback and will use get_dummies method
            categorical_cols = [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaymentMethod'
            ]
            for col in categorical_cols:
                if col not in raw.columns:
                    raw[col] = 'No'
            processed = pd.get_dummies(raw, columns=categorical_cols, drop_first=True)
            X = processed.values

        # Predict with base models
        lgb_prob = None
        deep_prob = None

        if lgbm_model is not None:
            try:
                lgb_prob = lgbm_model.predict_proba(X)[:, 1]
            except Exception:
                # sklearn warning about feature names difference - try re-shaping
                lgb_prob = lgbm_model.predict_proba(X)[:, 1]

        if deep_model is not None:
            deep_prob = deep_model.predict(X).ravel()

        # Decide which model's probability to use as the "chosen" score.
        # Default behavior: prefer stack (if available and both base probs exist),
        # otherwise prefer the model with the best F1 score according to models/metrics.json.
        chosen_prob = None
        used_model = None

        # Helper to safely read metrics.json and return best model by f1
        def pick_best_model_from_metrics():
            metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
            try:
                with open(metrics_path, 'r', encoding='utf-8') as mf:
                    mets = json.load(mf)
            except Exception:
                return None

            # Candidate keys we expect in metrics.json
            candidates = {}
            # Map known keys to our internal names
            mapping = {
                'stack': 'stack',
                'lightgbm': 'lightgbm',
                'lgbm': 'lightgbm',
                'deep_nn': 'deep_nn',
                'deep': 'deep_nn',
                'jupyter_mlp': 'jupyter_mlp'
            }

            for k, v in mets.items():
                mapped = mapping.get(k, None)
                if mapped is None:
                    continue
                # require f1 present and numeric
                f1 = None
                try:
                    f1 = float(v.get('f1', None))
                except Exception:
                    f1 = None
                if f1 is not None:
                    candidates[mapped] = max(candidates.get(mapped, 0), f1)

            if not candidates:
                return None

            # Prefer stack if present
            if 'stack' in candidates:
                return 'stack'

            # Otherwise pick highest f1
            best = max(candidates.items(), key=lambda x: x[1])
            return best[0]

        # If we have stack meta and both base probs, compute stack probability
        stack_prob = None
        if stack_meta is not None and lgb_prob is not None and deep_prob is not None:
            try:
                meta_X = np.vstack([lgb_prob, deep_prob]).T
                stack_prob = stack_meta.predict_proba(meta_X)[:, 1][0]
            except Exception:
                stack_prob = None

        # Choose model based on metrics.json preference
        preferred = pick_best_model_from_metrics()

        if preferred == 'stack' and stack_prob is not None:
            chosen_prob = float(stack_prob)
            used_model = 'stack'
        elif preferred == 'lightgbm' and lgb_prob is not None:
            chosen_prob = float(lgb_prob[0])
            used_model = 'lightgbm'
        elif preferred == 'deep_nn' and deep_prob is not None:
            chosen_prob = float(deep_prob[0])
            used_model = 'deep_nn'
        else:
            # Fallback behavior: stack if available, else lgbm, else deep
            if stack_prob is not None:
                chosen_prob = float(stack_prob)
                used_model = 'stack'
            elif lgb_prob is not None:
                chosen_prob = float(lgb_prob[0])
                used_model = 'lightgbm'
            elif deep_prob is not None:
                chosen_prob = float(deep_prob[0])
                used_model = 'deep_nn'
            else:
                return jsonify({'error': 'No models available for prediction.'}), 500

        churn_prob_percent = round(chosen_prob * 100, 2)
        THRESHOLD = 0.60
        result_text = 'Önleyici aksiyon alınmalı. Yüksek Terk Riski.' if chosen_prob >= THRESHOLD else 'Terk Riski Düşük.'

        # Also include per-model probabilities when available
        resp = {
            'churn_probability': f'{churn_prob_percent}%',
            'result_text': result_text,
            'used_model': used_model
        }
        try:
            if lgb_prob is not None:
                resp['lgb_probability'] = f"{round(float(lgb_prob[0]) * 100, 2)}%"
        except Exception:
            pass
        try:
            if deep_prob is not None:
                resp['deep_probability'] = f"{round(float(deep_prob[0]) * 100, 2)}%"
        except Exception:
            pass
        try:
            if stack_meta is not None and lgb_prob is not None and deep_prob is not None:
                resp['stack_probability'] = f"{round(float(stack_prob) * 100, 2)}%"
        except Exception:
            pass

        return jsonify(resp)

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400


if __name__ == '__main__':
    app.run(debug=True)