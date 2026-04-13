# MetroPT-3 Kestirimci Bakım Projesi

## Proje Hakkında

Porto Metro hava kompresörü sensör verisi kullanılarak anomali tespiti yapan makine öğrenmesi projesi.
7 aylık (Şubat-Eylül 2020) 1.5 milyon sensör kaydı analiz edildi.

## Kullanılan Teknolojiler

- **Veri işleme:** Polars, PyArrow
- **Görselleştirme:** Matplotlib, Seaborn
- **Modelleme:** Scikit-learn, TensorFlow/Keras
- **Açıklanabilirlik:** SHAP

## Proje Mimarisi

- **Lokal (Mac M3):** Veri ön işleme, EDA, raporlama
- **Kaggle (GPU T4):** Özellik mühendisliği, model eğitimi

## Yol Haritası

| Faz   | Açıklama                                            | Ortam  |
| ----- | ----------------------------------------------------- | ------ |
| Faz 1 | Veri optimizasyonu (CSV → Parquet, %92.5 küçülme) | Lokal  |
| Faz 2 | Ön işleme, oturum tespiti, etiketleme               | Lokal  |
| Faz 3 | Keşifçi veri analizi (EDA)                          | Lokal  |
| Faz 4 | Özellik mühendisliği (207 → 133 özellik)         | Kaggle |
| Faz 5 | Model eğitimi (IF, OC-SVM, LSTM Autoencoder)         | Kaggle |
| Faz 6 | SHAP analizi, iş metrikleri, raporlama               | Kaggle |

## Model Sonuçları

| Model            | Normal F1 | Şüpheli F1 | Yanlış Alarm |
| ---------------- | --------- | ------------ | -------------- |
| Isolation Forest | 0.93      | 0.08         | 92,504         |
| One-Class SVM    | 0.93      | 0.08         | 96,330         |
| LSTM Autoencoder | 0.94      | 0.07         | 70,336         |

## Veri Seti

[MetroPT-3 Dataset — Kaggle](https://www.kaggle.com/datasets/anshtanwar/metro-train-dataset)

# 🏭 Endüstriyel Kompresörlerde Kestirimci Bakım: Erken Uyarı ve Kök Neden Analizi

Bu proje, sensör verileriyle izlenen endüstriyel bir hava kompresöründe meydana gelen mekanik arızaları **yaşanmadan önce tahmin etmeyi** (Predictive Maintenance) ve bu arızaların **kök nedenlerini (SHAP)** tespit etmeyi amaçlayan kapsamlı bir Derin Öğrenme (Deep Learning) çalışmasıdır.

## 🚀 Proje Vizyonu ve Dönüşümü

Proje başlangıçta gözetimsiz (unsupervised) bir Anomali Tespiti projesi olarak başlamış, ancak Keşifçi Veri Analizi (EDA) ve özel filtreleme teknikleri sayesinde **Gözetimli bir Erken Uyarı Sistemine (Supervised Early Warning System)** evrilmiştir.

### 🔍 Faz 1: Anomali Tespiti ve Gürültü Filtreleme (Tamamlandı)

Makinenin 1.5 milyon satırlık geçmiş sensör verisi üzerinde "Autoencoder" mimarisi kullanılarak bir yeniden yapılandırma (reconstruction) hatası (MSE) hesaplandı.

* **Sorun:** Model, kompresörün normal ritmik dalgalanmalarını (yalancı alarmları) arıza sanarak "False Positive" çöplüğü yarattı.
* **Mühendislik Çözümü (MAD Filtresi):** Yalancı alarmları ezmek ve sadece fiziksel krizleri izole etmek için Medyan Mutlak Sapma (MAD - Median Absolute Deviation) tabanlı 5x çarpanlı dinamik bir eşik (Threshold) filtresi geliştirildi.
* **Sonuç:** 1.5 milyon satırlık veriden, makinenin gerçekten fiziksel olarak teklediği **167 Kesin Arıza Olayı (Kriz Anı)** %100 doğrulukla izole edildi.

### 🕵️‍♂️ Kök Neden Analizi: TP2 Ritim Bozukluğu

İzole edilen bu 167 kriz anına **SHAP (SHapley Additive exPlanations)** sarmalayıcısı uygulandı.

* Sensörlerin çıplak gözle yapılan "Geniş Açı" (10.000 adımlık) analizinde, **TP2 sensörünün** kompresörün nefes alışverişini (basınç döngüsünü) temsil ettiği keşfedildi.
* 167 arızanın tamamının, makinenin 1.0 basınç seviyesine ulaşamayıp 0.5 seviyelerinde "boğulması" ve ritminin kırılması sonucu yaşandığı matematiksel ve fiziksel olarak kanıtlandı.

### 🔮 Faz 2: LSTM ile Erken Uyarı Sistemi (Aktif Geliştirme)

Artık arızaların ne zaman ve nasıl yaşandığını bildiğimiz için (Y Etiketleri), hedefimiz makineyi arıza anında değil, **arıza yaşanmadan 5 dakika (300 adım) önce** uyaracak bir kahin model kurmaktır.

* **Hedef:** Krizden önceki 300 adımı "Tehlike Bölgesi (1)" olarak etiketleyerek Sınıflandırma (Binary Classification) yapmak.
* **Mimari:** Zaman serilerindeki hafıza yeteneğinden dolayı Gözetimli **LSTM (Long Short-Term Memory)** ağı kullanılmaktadır.
* **Odak:** Klasik doğruluk (Accuracy) yerine, "Gerçekleşecek arızaların kaçını önceden bilebildik?" sorusuna yanıt veren **Recall (Duyarlılık)** metriği maksimize edilecektir.

## 🛠️ Kullanılan Teknolojiler

* **Veri İşleme:** Pandas, Numpy, Polars (Büyük veri optimizasyonu)
* **Makine Öğrenmesi & Derin Öğrenme:** TensorFlow, Keras (Autoencoder & LSTM)
* **Açıklanabilir Yapay Zeka (XAI):** SHAP (GradientExplainer)
* **Görselleştirme:** Matplotlib (Fiziksel sensör ritim grafikleri)

## 📂 Proje Adımları

- [X] Veri Temizleme ve Ölçeklendirme (MinMaxScaler)
- [X] Autoencoder ile Gözetimsiz Anomali Skoru (MSE) Üretimi
- [X] MAD İstatistiksel Filtresi ile Yalancı Alarmların (False Positives) Temizlenmesi
- [X] SHAP ile Kök Neden (Root Cause) Analizi
- [X] Fiziksel Sensör Doğrulaması (TP2 Ritim Bozukluğu Tespiti)
- [X] Pre-Anomaly (Erken Uyarı) Y Etiketlerinin Oluşturulması ve Veri Paketi İhracı
- [ ] *[Yeni Notebook]* Zaman Serisi Train/Test Bölünmesi (Chronological Split)
- [ ] *[Yeni Notebook]* Predictive LSTM Modelinin İnşası ve Eğitimi
- [ ] *[Yeni Notebook]* Recall Odaklı Eşik Optimizasyonu ve Test
