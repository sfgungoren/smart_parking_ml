AI-powered smart parking prediction system using machine learning and real-time adjustments.

# Akıllı Otopark Tahmin Sistemi
### Smart Parking Prediction System

Bu proje, makine öğrenmesi ve gerçek zamanlı veri düzeltme algoritmalarını birleştirerek otopark doluluk oranlarını tahmin eden **hibrit** bir sistemdir.

---

## 🚀 Öne Çıkan Özellikler

- **🧠 Hibrit Tahmin Motoru**: 
  - **Random Forest**: Geçmiş trendlere dayalı temel tahmin.
  - **Gaussian Event Impact**: Özel etkinliklerin (maç, konser vb.) otopark yoğunluğuna etkisini zaman tabanlı modeller.
  - **Exponential Decay**: Gerçek zamanlı doluluk verisi ile tahmin arasındaki farkı anlık olarak düzeltir.
- **📊 Modern Dashboard**: Hem kullanıcılar hem de yöneticiler için geliştirilmiş şık ve responsive arayüz.
- **⚙️ Dinamik Yönetim**: Etkinlik ekleme/silme ve anlık otopark durumunu manuel simüle etme imkanı.

---

## 📦 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1. **Projeyi Klonlayın**:
   ```bash
   git clone https://github.com/kullaniciadi/smart-parking-ml.git
   cd smart-parking-ml
   ```

2. **Bağımlılıkları Yükleyin**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Modeli Eğitin (Önemli)**:
   Sistem varsayılan bir modelle gelmez. Tam performans için modelin eğitilmesi gerekir:
   ```bash
   python train_model.py
   ```
   *Model yoksa sistem otomatik olarak "Fallback" modunda (varsayılan değerlerle) çalışmaya devam eder.*

4. **Uygulamayı Başlatın**:
   ```bash
   python app.py
   ```
   Tarayıcınızdan `http://127.0.0.1:5000` adresine giderek sistemi kullanmaya başlayabilirsiniz.

---

## ⚙️ Sistem Nasıl Çalışır?

Tahmin motoru üç aşamalı bir mantıkla çalışır:
1. **Temel ML Tahmini**: Model; saat, gün ve geçmiş doluluk trendlerini analiz eder.
2. **Etkinlik Etkisi**: Tanımlanan etkinliklerin yoğunluğu Gaussian dağılımı (çan eğrisi) kullanılarak hesaplanır.
3. **Gerçek Zamanlı Düzeltme**: Otoparkın o anki gerçek doluluğu ile tahmin arasındaki fark, zamanla sönümlenerek (exponential decay) geleceğe yansıtılır.

---

## 📂 Proje Yapısı

- `app.py`: Ana Flask sunucusu ve tahmin motoru mantığı.
- `train_model.py`: Model eğitim ve backtest betiği.
- `templates/`: HTML arayüz dosyaları.
- `static/`: CSS ve stil dosyaları.
- `tools/`: 🧪 Veri analizi ve görselleştirme için yardımcı scriptler (Saatlik doluluk analizi vb.) içerir.
- `data.json`: Sistemin anlık durumunu ve etkinlikleri saklar. İlk çalıştırmada otomatik olarak oluşturulur.

---

## 📊 Veri Seti (Dataset)

Projede kullanılan `dataset.csv`, Birmingham şehri otopark verilerini içeren halka açık bir veri setidir.

---
*Bu proje, akıllı şehir teknolojileri ve veri bilimi uygulamaları kapsamında geliştirilmiştir.*
