📊 Telco Müşteri Terk (Churn) Analizi Projesi
Telekomünikasyon sektöründe müşteri kaybını minimize etmek için geliştirilmiş, uçtan uca makine öğrenmesi ve interaktif yönetim panelini içeren bir çözümdür. Ham veriden tahmine kadar tüm süreç modüler bir yapıda kurgulanmıştır.

🎯 Projenin Amacı
Müşterilerin abonelik iptal etme olasılıklarını önceden tahmin ederek; özel kampanyalar ve indirimler gibi proaktif önlemler alınmasını sağlamaktır.

🛠️ Teknik Mimari ve Model Stratejisi
Projede yüksek doğruluk için hibrit bir modelleme yaklaşımı benimsenmiştir:

LightGBM: Optuna ile hiperparametre optimizasyonu yapılmış, hızlı ve yüksek performanslı gradyan artırma algoritması.

Derin Öğrenme (MLP): Keras tabanlı, BatchNormalization ve Dropout katmanlarıyla normalize edilmiş Çok Katmanlı Algılayıcı.

OOF Stacking: Modellerin tahminlerini birleştiren Logistic Regression tabanlı meta-model.

SMOTE: Veri setindeki dengesiz sınıf dağılımını (terk eden müşteriler) yönetmek için kullanılmıştır.

💻 Teknoloji Yığını
Backend: Python, Flask

Frontend: HTML5, CSS3, Jinja2, Chart.js

Veri Bilimi: Scikit-learn, Pandas, TensorFlow/Keras, Optuna, Joblib

📂 Dosya Yapısı
app.py: Model yükleme, API yönetimi ve web arayüzü kontrol merkezi.

train_optuna_stack.py: Optuna tuning ve Stacking model eğitim scripti.

models/: Kayıtlı modeller (.pkl, .h5) ve ön işleme nesneleri.

models/metrics.json: Modellerin başarı kriterlerini (F1, Accuracy vb.) tutan dinamik veri dosyası.

⚙️ Karar Mekanizması
Girdi: Kullanıcı verileri arayüz üzerinden girer.

Seçim: Sistem, metrics.json içindeki en yüksek F1 skoruna sahip modeli otomatik seçer.

Tahmin: Seçilen modelin ürettiği olasılık %60 (0.60) eşiğini aşarsa "Yüksek Terk Riski" uyarısı tetiklenir.

🚀 Kurulum ve Çalıştırma
PowerShell

1. Sanal Ortam Oluşturma: 
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2. Bağımlılıkları Yükleme: 
pip install -r requirements.txt

3.Uygulamayı çalıştırma: 
flask run
Tarayıcıda http://127.0.0.1:5000 adresinden panele ulaşabilirsiniz.

🔮 Gelecek Yol Haritası
Olasılık Kalibrasyonu: Platt Scaling ile tahmin güvenilirliğini artırmak.

MLflow Entegrasyonu: Model deneylerini daha sistematik takip etmek.

Dinamik Eşikler: Risk iştahına göre eşik değerini UI üzerinden ayarlama özelliği.


