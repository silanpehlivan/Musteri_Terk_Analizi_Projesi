<div align="center">

# Telco Churn Analysis

**Müşteri terk riski analizi**

![Python](https://img.shields.io/badge/Python-2563eb?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-0891b2?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-7c3aed?style=flat-square)
[![MIT License](https://img.shields.io/badge/License-MIT-16a34a?style=flat-square)](LICENSE)

Telekomünikasyon müşteri verilerinden terk olasılığını tahmin eden, model karşılaştırması ve web paneli içeren makine öğrenmesi projesi.

</div>

---

## Öne Çıkanlar

- Optuna ile LightGBM optimizasyonu
- MLP ve OOF stacking model yaklaşımı
- F1 skoruna göre model seçimi ve risk gösterimi

## Teknolojiler

Python · Flask · LightGBM · Keras

<details>
<summary><strong>Kurulum, kullanım ve teknik ayrıntılar</strong></summary>

Telekomünikasyon sektöründe müşteri kaybını minimize etmek için geliştirilmiş, uçtan uca makine öğrenmesi ve interaktif yönetim panelini içeren bir çözümdür. Ham veriden tahmine kadar tüm süreç modüler bir yapıda kurgulanmıştır.

## Projenin Amacı

Müşterilerin abonelik iptal etme olasılıklarını önceden tahmin ederek; özel kampanyalar ve indirimler gibi proaktif önlemler alınmasını sağlamaktır.

## Teknik Mimari ve Model Stratejisi

Projede yüksek doğruluk için hibrit bir modelleme yaklaşımı benimsenmiştir:

- **LightGBM:** Optuna ile hiperparametre optimizasyonu yapılmış, hızlı ve yüksek performanslı gradyan artırma algoritması.

- **Derin Öğrenme (MLP):** Keras tabanlı, BatchNormalization ve Dropout katmanlarıyla normalize edilmiş Çok Katmanlı Algılayıcı.

- **OOF Stacking:** Modellerin tahminlerini birleştiren Logistic Regression tabanlı meta-model.

- **SMOTE:** Veri setindeki dengesiz sınıf dağılımını (terk eden müşteriler) yönetmek için kullanılmıştır.

## Teknoloji Yığını

### Backend
- Python
- Flask

### Frontend
- HTML5
- CSS3
- Jinja2
- Chart.js

### Veri Bilimi
- Scikit-learn
- Pandas
- TensorFlow/Keras
- Optuna
- Joblib

## Dosya Yapısı

- **app.py:** Model yükleme, API yönetimi ve web arayüzü kontrol merkezi.

- **train_optuna_stack.py:** Optuna tuning ve Stacking model eğitim scripti.

- **models/:** Kayıtlı modeller (.pkl, .h5) ve ön işleme nesneleri.

- **models/metrics.json:** Modellerin başarı kriterlerini (F1, Accuracy vb.) tutan dinamik veri dosyası.

## Karar Mekanizması

- **Girdi:** Kullanıcı verileri arayüz üzerinden girer.

- **Seçim:** Sistem, metrics.json içindeki en yüksek F1 skoruna sahip modeli otomatik seçer.

- **Tahmin:** Seçilen modelin ürettiği olasılık %60 (0.60) eşiğini aşarsa "Yüksek Terk Riski" uyarısı tetiklenir.

## Kurulum ve Çalıştırma

### PowerShell

#### 1. Sanal Ortam Oluşturma

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### 2. Bağımlılıkları Yükleme

```powershell
pip install -r requirements.txt
```

#### 3. Uygulamayı Çalıştırma

```powershell
flask run
```

Tarayıcıda aşağıdaki adres üzerinden panele ulaşabilirsiniz:

```text
http://127.0.0.1:5000
```

## Gelecek Yol Haritası

- **Olasılık Kalibrasyonu:** Platt Scaling ile tahmin güvenilirliğini artırmak.

- **MLflow Entegrasyonu:** Model deneylerini daha sistematik takip etmek.

- **Dinamik Eşikler:** Risk iştahına göre eşik değerini UI üzerinden ayarlama özelliği.


</details>

---

<div align="center">

**© 2026 Şilan PEHLİVAN**

Bu proje MIT lisansı kapsamında sunulmaktadır. Kullanım ve dağıtım koşulları: [LICENSE](LICENSE).

</div>
