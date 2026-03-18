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
