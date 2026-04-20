# 🔧 MetroPT-3 Kompresör Erken Arıza Uyarı Sistemi

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/AI_Engine-TensorFlow%2FKeras-orange?style=for-the-badge&logo=tensorflow)
![Polars](https://img.shields.io/badge/Data-Polars-blue?style=for-the-badge&logo=polars)
![Kaggle](https://img.shields.io/badge/Training-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)
![SHAP](https://img.shields.io/badge/XAI-SHAP-blueviolet?style=for-the-badge)

**BU PROJE UÇTAN UCA BİR MAKİNE ÖĞRENMESİ VE VERİ MÜHENDİSLİĞİ ÇALIŞMASIDIR**

Proje, Portekiz metro sisteminin gerçek pnömatik kompresör sensör verisini (MetroPT-3) analiz ederek, makineyi durduracak bir arızanın yaşanmasından **5 dakika önce uyarı** veren bir yapay zeka sistemi geliştirmektedir.

Sadece bir Jupyter Notebook analizi değil;  **Veri Optimizasyonu** ,  **Denetimsiz Anomali Tespiti (LSTM Autoencoder)** ,  **Açıklanabilir Yapay Zeka (SHAP)** , **Denetimli Erken Uyarı (LSTM Classifier)** ve **Canlı Çıktı (MVP Fonksiyonu)** katmanlarının birbirini besleyerek ilerlediği gerçek bir mühendislik sürecidir.

---

## 🌐 Özellikler

### 🚀 Model Geliştirme Süreci (Engineering Journey)

Projenin hikâyesi, temiz bir etiketten çok daha zorlu bir yerden başlar: **Etiket Yokluğu.** Elimizde yalnızca ham sensör verileri vardı; arızaların tam olarak ne zaman gerçekleştiğine dair tek bir kayıt bile bulunmuyordu. Bu durum, geliştirme sürecini aşağıdaki aşamalara yöneltti:

1. **Veri Mühendisliği (Baseline):** Ham 208 MB CSV verisi, Polars ile işlenerek 15.7 MB Parquet formatına dönüştürüldü (%92.5 küçülme). Sensör ölçeklemeleri optimize edildi.
2. **Kural Tabanlı Etiketleme (Heuristic Labeling):** Veri setindeki 331 oturuma bölündü; 12 saatten uzun oturumların son 2 saatlik dilimi `is_suspect=1` olarak işaretlendi.
3. **Denetimsiz Öğrenme (Unsupervised Baseline):** Isolation Forest ve One-Class SVM modelleri denendi. Her ikisi de F1 ≈ 0.08 ile başarısız oldu. Nedeni: etiketlerdeki devasa gürültü.
4. **LSTM Autoencoder:** Yalnızca normal veriyle eğitilen bir yeniden yapılandırma modeli kuruldu. Reconstruction Error (MSE) üzerinden 167 gerçek kriz anı tespit edildi.
5. **SHAP ile Kök Neden Analizi:** 127 sensör arasından TP2 (kompresör çıkış basıncı) tüm anomalilerin baş sorumlusu olarak belirlendi. Dört adet bozuk sensör (veri kirliliği kaynakları) temizlendi ve model sıfırdan yeniden eğitildi.
6. **Karar (Pivot - Denetimli Öğrenme):** Tespit edilen 167 kriz anı birer "çarpışma noktası" kabul edilerek her birinin 5 dakika (300 adım) öncesi `y=1 (Tehlike)` olarak etiketlendi. Bu **Hedef Kaydırma (Pre-Anomaly Labeling)** tekniği ile gerçek bir erken uyarı modeli eğitimine geçildi.
7. **Optimal Eşik (Threshold Optimization):** Precision-Recall eğrisi üzerinden F1 skoru maksimize edilerek optimal karar eşiği matematiksel olarak `0.7011` olarak belirlendi. Model bu haliyle MVP olarak paketlendi.

---

### 🧠 Özellik Mühendisliği (The "Brain" of the Model)

Model kararlarını güçlendirmek için 4 ana kategoride 207 ham özellik türetildi; korelasyon filtresi sonrasında 133 anlamlı özellik elde edildi.

#### 1. 🔁 KAYAN PENCERE İSTATİSTİKLERİ (Rolling Features)

*"Bu sensör son 1 saatte ortalama ne yapıyor, ne kadar salınım gösteriyor?"*

3 farklı zaman penceresinde (`1h`, `6h`, `24h`) her sensör için **ortalama** ve **standart sapma** hesaplandı.

* **`{sensor}_rmean_1h` (Kısa Bellek):** Son 1 saatin yumuşatılmış trendi. Anlık gürültüyü bastırır.
* **`{sensor}_rstd_6h` (Kararlılık):** Son 6 saatin standart sapması. Sensörün ne kadar titreştiğini ölçer.
* **`{sensor}_rmean_24h` (Uzun Hafıza):** Günlük baz çizgisi. Normdan uzaklaşmayı tespit eder.

> *Mühendislik Yorumu:* Tek bir anlık değer sadece o anki durumu söyler. Pencereleme ile modele "bu değer normalde böyle mi olur?" sorusunu sordururuz.

#### 2. ⏪ GECİKMELİ DEĞİŞİMler (Lag Features)

*"Bu sensör 10 dakika öncesine göre ne kadar değişti?"*

3 farklı gecikme adımında (`10min`, `1h`, `6h`) her sensör için hem **geçmiş değer** hem de **anlık değişim miktarı** hesaplandı.

* **`{sensor}_lag_10min` (Kısa Geçmiş):** Modelin 10 dakika öncesini hatırlaması.
* **`{sensor}_diff_1h` (Sapma Hızı):** 1 saatte ne kadar değişti? Yavaş yükselen basınç, hızlı düşen akım gibi örüntüler.

> *Senaryo:* Motor akımı son 1 saatte +2.3 A artmışsa, motor giderek daha fazla yük altına giriyor demektir. Model bu gradyanı anomali sinyali olarak öğrenebilir.

#### 3. 🔗 ÇAPRAZ SENSÖR ÖZELLİKLERİ (Cross-Sensor Features)

*"TP2 basıncı ile motor akımı aynı anda ne yapıyor?"*

Fiziksel ilişkileri bilen 4 yeni özellik türetildi:

* **`TP2_TP3_ratio` (Basınç Dengesi):** İki basınç sensörü oranı. Normalden saptığında hava kaçağı sinyali.
* **`motor_heat_index` (Çift Stres):** Motor akımı × Yağ sıcaklığı. Aynı anda yük ve ısı arttığında risk katlanır.
* **`pressure_diff_TP3_H1` (Sızıntı Göstergesi):** TP3 ile H1 arasındaki fark. Tıkanıklık veya sızıntı belirtisi.
* **`pressure_motor_load` (Kompresör Zorlanması):** Basınç × Akım. Sistem en ağır yük altındayken artar.

#### 4. 🎵 FREKANS ANALİZİ (FFT Features)

*"Motor akımının titreşim ritmi bozuldu mu?"*

`Motor_current` sensörü için son 1 saatlik pencerede **Hızlı Fourier Dönüşümü (FFT)** uygulandı.

* **`fft_dominant_freq` (Baskın Ritim):** Sistemin en güçlü titreşim frekansı.
* **`fft_dominant_power` (Titreşim Şiddeti):** O frekansın ne kadar enerjiye sahip olduğu.
* **`fft_total_power` (Toplam Titreşim Enerjisi):** Sistemin genel mekanik gürültüsü.

> *Mantık:* Doğrudan titreşim sensörümüz olmasa da motor akımı, kompresörün mekanik ritmini dolaylı olarak yansıtır. Normal bir motor belli bir frekansta döner; bu ritim bozulduğunda FFT bunu yakalar.

---

### 🔬 Teknik Mimari ve Optimizasyon

#### 📦 Polars ile Veri Optimizasyonu

Pandas'ın aksine Polars, **Lazy Evaluation** ve **çok çekirdekli işlem** kullanır. Float64 → Float32 dönüşümü ve kolonsal Parquet formatına yazma ile veri:

* **208.2 MB CSV → 15.7 MB Parquet** (%92.5 küçülme)
* Bunun sebebi kompresör sensörlerinin **Run-Length Encoding (RLE)** ile inanılmaz sıkışmasıdır; makine çoğu zaman aynı durumda çalışır, aynı değerler tekrar eder ve Parquet "bu değer 10.000 kez tekrar ediyor" diyerek sadece bir kez yazar.

#### 🔍 LSTM Autoencoder ile Denetimsiz Anomali Tespiti

Etiket olmadan anomali bulmak için **Encoder-Decoder** mimarisi kuruldu:

* Model yalnızca **normal verilerle** eğitilir; normalin nasıl göründüğünü öğrenir.
* Test aşamasında her giriş yeniden inşa edilir. **Reconstruction Error (MSE)** yüksekse model o anı tanıyamamış demektir → Anomali.
* Bozuk dört sensör (`DV_pressure_diff_10min`, `Pressure_switch_lag_10min`, `Towers_lag_1h`, `Towers_lag_6h`) SHAP analizi ile tespit edilerek temizlendi ve model 127 özellikle sıfırdan eğitildi.

#### 🧪 SHAP ile Açıklanabilir Yapay Zeka (XAI)

127 sensör arasından hangisinin anomaliye yol açtığını anlamak için **GradientExplainer** kullanıldı:

* SHAP'a LSTM'in 3 boyutlu çıktısı direkt verilemez; bunun için MSE hesaplayan bir **Sarmalayıcı Model (Wrapper)** yazıldı.
* Top 500 ekstrem anomali ve rastgele 100 kontrol testi, her iki SHAP grafiğinde de **TP2 (kompresör çıkış basıncı)** tek başına baskın çıktı.
* Bu sonuç fiziksel gerçekle örtüşmektedir: Metro pnömatik sistemlerindeki anomalilerin büyük çoğunluğu hava kaçakları ve kompresör kaynaklı arızalardır.

#### ⚡ Focal Loss ile Sınıf Dengesizliği Yönetimi

Veri setinde tehlike anları yalnızca ~%7 oranında. Standart Binary Crossentropy bu dengesizlikte "Her şey normaldir" diyerek %93 doğruluk elde eder, ancak bir arızayı bile yakalamaz. **Focal Loss** (`gamma=2.0, alpha=0.8`) kolay öğrenilen normal anlara verilen ağırlığı ezerken, zor ve az sayıdaki tehlike anlarına odaklanmayı zorlar.

#### 📐 Dinamik Eşik Optimizasyonu (Threshold Optimization)

Fabrikada iki tür hata farklı maliyete sahiptir:

* **False Negative (Kaçırılan Alarm):** Makine patlar → Felaket.
* **False Positive (Yalancı Alarm):** İşçiler boşuna koşuşturur → Güven kaybı.

Sabit %50 eşiği yerine, **Precision-Recall eğrisi** üzerinde F1 skoru maximize edilerek matematiksel optimum `0.7011` olarak hesaplandı. Bu eşikte model %70 emin olmadan alarm çalmaz; alarm çaldığında gerçek kriz ihtimali %65'e yükselir.

---

## 🛠️ Mimari ve Teknolojiler

| Dosya                                    | Görev                                                   | Kullanılan Teknolojiler                                                  |
| :--------------------------------------- | :------------------------------------------------------- | :------------------------------------------------------------------------ |
| **01_data_optimization.ipynb**     | Ham CSV → Optimized Parquet Dönüşümü               | `Polars`,`Parquet`,`Float32 Casting`                                |
| **02_preprocessing.ipynb**         | Oturum Tespiti, Şüpheli Pencere Etiketleme             | `Polars`,`Session Segmentation`,`Heuristic Labeling`                |
| **03_eda.ipynb**                   | Sensör Dağılımları, Korelasyon ve Örüntü Analizi | `Matplotlib`,`Seaborn`,`MinMaxScaler`                               |
| **04_kaggle.ipynb**                | Rolling, Lag, FFT, Çapraz Sensör Özellik Üretimi     | `Polars`,`NumPy`,`FFT`                                              |
| **05_kaggle_model_training.ipynb** | Denetimsiz Modeller + LSTM Autoencoder Eğitimi          | `Scikit-learn`,`TensorFlow/Keras`,`IsolationForest`,`OneClassSVM` |
| **06_kaggle_reporting.ipynb**      | SHAP Analizi, Bozuk Sensör Tespiti, Dinamik Eşik       | `SHAP`,`GradientExplainer`,`MAD Statistics`                         |
| **07_predictive_LSTM_model.ipynb** | Denetimli Erken Uyarı Modeli, MVP Paketi                | `TensorFlow/Keras`,`Focal Loss`,`Recall Optimization`               |

---

## 📂 Proje Dizini

```bash
metropt3-compressor-predictive-maintenance/
├── notebooks/
│   ├── 01_data_optimization.ipynb     # Ham Veri Optimizasyonu
│   ├── 02_preprocessing.ipynb         # Oturum Tespiti ve Etiketleme
│   ├── 03_eda.ipynb                   # Keşifsel Veri Analizi
│   ├── 04_kaggle.ipynb                # Özellik Mühendisliği (Kaggle)
│   ├── 05_kaggle_model_training.ipynb # Denetimsiz Model Eğitimleri (Kaggle)
│   ├── 06_kaggle_reporting.ipynb      # SHAP Analizi ve Raporlama (Kaggle)
│   └── 07_predictive_LSTM_model.ipynb # Denetimli Erken Uyarı LSTM (Kaggle)
├── data/
│   ├── raw/
│   │   └── MetroPT3(AirCompressor).csv        # Ham Veri (208 MB)
│   └── processed/
│       ├── metropt3_optimized.parquet          # Tip Optimize Edilmiş Veri (15.7 MB)
│       ├── metropt3_labeled.parquet            # is_suspect Etiketli Veri
│       └── metropt3_features.parquet           # 133 Özellikli Nihai Veri Seti
├── models/
│   ├── lstm_autoencoder.keras                  # Denetimsiz Anomali Dedektörü
│   ├── kompresor_erken_uyari_v1_MVP.keras      # Erken Uyarı Sınıflandırıcısı (MVP)
│   ├── yeni_scaler_127.pkl                     # Özellik Ölçekleyici
│   └── lstm_threshold.npy                      # Autoencoder Eşik Değeri
├── reports/
│   └── figures/
│       ├── 01_timeseries_overview.png
│       ├── 02_daily_overview.png
│       ├── 03_distributions.png
│       ├── 05_correlation_heatmap.png
│       ├── 06_patterns.png
│       └── 07_monthly_heatmap.png
└── README.md
```

---

## 🔬 Tahmin Metodolojisi (Nasıl Çalışıyor?)

Sistem, canlı ortamda şu 4 aşamalı süreci işletir:

1. **Veri Alımı (Data Ingestion):**
   Kompresöre bağlı 15 sensörden her 10 saniyede bir yeni ölçüm alınır. Son 30 adım (yaklaşık 5 dakika) bir **zaman penceresi** oluşturur.
2. **Özellik Dönüşümü (On-the-fly Feature Engineering):**
   Anlık sensör değerleri; kayan ortalamalar, gecikmeli değişimler ve çapraz sensör oranları ile **127 özellikli bir vektöre** dönüştürülür. Eğitimde kullanılan `MinMaxScaler` ile normalize edilir.
3. **Model Sorgulama (Inference):**
   Hazırlanan `(1, 30, 127)` boyutlu tensör, `kompresor_erken_uyari_v1_MVP.keras` modeline beslenir. Model, `0.00` ile `1.00` arasında bir **Tehlike Olasılığı** üretir.
4. **Karar ve Sunum:**
   * **Optimal Eşik:** `0.7011` olasılığın altındaki durumlar "Normal" olarak geçer.
   * Eşiği aşan durumlarda `🚨 DİKKAT! Ritim bozukluğu tespit edildi!` uyarısı yayınlanır.
   * Model, arızadan **5 dakika önce** uyarıyı üretecek şekilde etiketlenmiş veriyle eğitildiğinden müdahale penceresi sağlanmış olur.

---

## 🔬 MVP Performans Özeti

| Metrik                          | Değer                                                     |
| :------------------------------ | :--------------------------------------------------------- |
| **Optimal Eşik**         | 0.7011                                                     |
| **Tehlike - Precision**   | %65 (alarm çaldığında 3'te 2'si gerçek)               |
| **Tehlike - Recall**      | %53 (gerçek krizlerin yarısından fazlası yakalanıyor) |
| **Normal - Recall**       | %95 (makine sağlamken model %95 sessiz kalıyor)          |
| **Yalancı Alarm Oranı** | %5 (normal çalışmada gereksiz alarm)                    |

> ⚠️ **MVP Notu:** Bu sonuçlar denetimsiz aşamada bulunan 167 kriz anı üzerinden elde edilmiştir. Gerçek etiketli bir veri seti veya daha uzun eğitim verisiyle modelin Recall değerinin önemli ölçüde artması beklenmektedir. Mevcut haliyle demo ve prototip kurulum için uygundur.

---

## ⚙️ Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install polars pandas numpy scikit-learn tensorflow shap matplotlib seaborn
```

### Adım 1: Veri Optimizasyonu

```bash
jupyter notebook notebooks/01_data_optimization.ipynb
```

Ham CSV veriyi Parquet formatına çevirir.

### Adım 2: Ön İşleme ve EDA

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_eda.ipynb
```

Oturumları tespit eder, şüpheli dönemleri etiketler ve keşifsel analizi çalıştırır.

### Adım 3: Özellik Mühendisliği ve Model Eğitimi (Kaggle)

`04_kaggle.ipynb`, `05_kaggle_model_training.ipynb`, `06_kaggle_reporting.ipynb` dosyaları Kaggle ortamında GPU ile çalıştırılmak üzere tasarlanmıştır. İşlenmiş `metropt3_labeled.parquet` dosyasını Kaggle dataset olarak yükleyip sırayla çalıştırınız.

### Adım 4: Erken Uyarı Modeli (MVP)

```bash
jupyter notebook notebooks/07_predictive_LSTM_model.ipynb
```

Kaggle'dan indirilen `lstm_hazirlik_verisi.npz` dosyası ile denetimli erken uyarı modelini eğitir ve MVP fonksiyonunu çalıştırır.

### Canlı Kullanım (MVP Fonksiyonu)

```python
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/kompresor_erken_uyari_v1_MVP.keras')
OPTIMAL_ESIK = 0.7011

def kompresor_durumu_kontrol_et(anlik_sensor_verisi_30_adim):
    if len(anlik_sensor_verisi_30_adim.shape) == 2:
        anlik_sensor_verisi_30_adim = np.expand_dims(anlik_sensor_verisi_30_adim, axis=0)
    tehlike_ihtimali = model.predict(anlik_sensor_verisi_30_adim, verbose=0)[0][0]
    if tehlike_ihtimali >= OPTIMAL_ESIK:
        print(f"🚨 DİKKAT! Ritim bozukluğu tespit edildi! (Eminlik: %{tehlike_ihtimali*100:.1f})")
        print("-> Lütfen 5 dakika içinde makineyi kontrol edin.")
    else:
        print(f"✅ Sistem Normal. (Tehlike İhtimali: %{tehlike_ihtimali*100:.1f})")

# 30 adım × 127 özellik boyutunda normalize edilmiş vektörü ver
kompresor_durumu_kontrol_et(sensor_verisi)
```

---

## 🤝 Katkıda Bulunma

Pull request'ler kabul edilir. Büyük değişiklikler veya özellik önerileri için lütfen önce "Issues" bölümünde tartışma başlatın.

## 👤 İletişim

Bu proje **Deniz BAYAT** tarafından geliştirilmiştir.

* **LinkedIn:** [linkedin.com/in/denizbayat1/](https://www.linkedin.com/in/denizbayat1/)
* **GitHub:** [github.com/1DenBay](https://github.com/1DenBay)
* **Medium:** [medium.com/@denizbyat](https://medium.com/@denizbyat)
* **Email:** denizbyat@gmail.com
