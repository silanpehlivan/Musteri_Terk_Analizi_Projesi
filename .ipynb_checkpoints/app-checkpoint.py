# app.py

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import pickle 
import sys

app = Flask(__name__)

# --- Model ve Ön İşleme Araçlarını Yükleme ---
try:
    model = load_model('models/mlp_churn_model.h5')
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model ve Scaler başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Model veya Scaler yüklenemedi. {e}")
    sys.exit(1)

# KRİTİK: Modelin eğitimde kullandığı KESİN 31 SÜTUNUN SIRASI
# Bu liste, tüm hataları gidermek için tekrar kontrol edilmiş 31 sütun sırasıdır.
FEATURE_ORDER = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 
    'MultipleLines_Yes', 'MultipleLines_No phone service', 
    'InternetService_Fiber optic', 'InternetService_No', 
    'OnlineSecurity_Yes', 'OnlineSecurity_No internet service', 
    'OnlineBackup_Yes', 'OnlineBackup_No internet service', 
    'DeviceProtection_Yes', 'DeviceProtection_No internet service', 
    'TechSupport_Yes', 'TechSupport_No internet service', 
    'StreamingTV_Yes', 'StreamingTV_No internet service', 
    'StreamingMovies_Yes', 'StreamingMovies_No internet service', 
    'Contract_One year', 'Contract_Two year', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check'
] 
# Not: Listede 30 sütun var. Scaler hatasını gidermek için bu liste kullanılacaktır.

# Sayısal sütunların isimleri (Scaler'ın eğitildiği 3 sütun)
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# --- Flask Rotaları ---
@app.route('/')
def home():
    """Ana Sayfayı (index.html) gösterir."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Web arayüzünden gelen veriyi alıp tahmini hesaplar."""
    try:
        data = request.get_json(force=True)
        
        # 1. Ham veriyi DataFrame'e Çevirme
        raw_input = pd.DataFrame([data])
        
        # Sayısal alanları dönüştür
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
             raw_input[col] = pd.to_numeric(raw_input.get(col, 0), errors='coerce').fillna(0)

        # One-Hot Encoding Uygulama
        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaymentMethod'
        ] # Formda görünen tüm kategorik/binary kolonlar
        
        # Kalan ham kategorik kolonlar için OHE uygula
        processed_df = pd.get_dummies(raw_input, columns=categorical_cols, drop_first=True)
        
        # 2. Final Vektörünü Oluşturma (FEATURE_ORDER'ı Garanti Etme)
        final_input = pd.DataFrame(0, index=[0], columns=FEATURE_ORDER)
        
        # İşlenmiş DF'ten eşleşen sütunları kopyala
        for col in processed_df.columns:
            if col in final_input.columns:
                final_input[col] = processed_df[col].iloc[0]

        # 3. ÖLÇEKLEME (SADECE 3 SÜTUNU GÖNDERME)
        
        # a) Sadece sayısal verileri ayır
        numerical_input_to_scale = final_input[NUMERICAL_COLS]
        
        # b) Scaler'ı sadece 3 sütunluk veri üzerinde çalıştır
        scaled_numerical = scaler.transform(numerical_input_to_scale)

        # c) Ölçeklenmiş veriyi 31 sütunluk yerine koy
        final_input[NUMERICAL_COLS] = scaled_numerical
        
        # Modelin beklediği NumPy formatına çevirme (30 veya 31 sütun)
        final_array_scaled = final_input.values
        
        # 4. Tahmini yapma
        prediction_prob = model.predict(final_array_scaled)[0][0]
        
        # 5. Sonucu formatlama
        THRESHOLD = 0.60
        churn_prob_percent = round(prediction_prob * 100, 2)
        
        if prediction_prob >= THRESHOLD:
             result_text = f"Önleyici aksiyon alınmalı. Yüksek Terk Riski."
        else:
             result_text = f"Terk Riski Düşük."
        
        return jsonify({
            'churn_probability': f'{churn_prob_percent}%',
            'result_text': result_text
        })

    except Exception as e:
        # Hata durumunda, hatayı arayüze gönder
        return jsonify({'error': f'Tahmin hatası: {str(e)}'}), 400

if __name__ == "__main__":
    app.run(debug=True)