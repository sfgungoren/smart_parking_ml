from flask import Flask, render_template, jsonify, request
import json
import os
import joblib
import pandas as pd

from datetime import datetime, timedelta
import math

app = Flask(__name__)

# Yapılandırma
DATA_FILE = "data.json"
MODEL_PATH = "parking_model.joblib"
META_PATH = "model_metadata.joblib"
CAPACITY = 100

def load_data():
    """Sistem verilerini data.json dosyasından yükler. Dosya yoksa varsayılan değerlerle oluşturur."""
    if not os.path.exists(DATA_FILE):
        default_data = {
            "capacity": CAPACITY,
            "current_occupancy": 0,
            "slots": [False] * CAPACITY,
            "events": []
        }
        save_data(default_data)
        return default_data
    
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"Hata: {DATA_FILE} okunurken sorun oluştu. Varsayılan veriler yükleniyor.")
        return {"capacity": CAPACITY, "current_occupancy": 0, "slots": [False] * CAPACITY, "events": []}


def save_data(data):
    """Sistem verilerini data.json dosyasına kaydeder."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

class PredictionEngine:
    """Hibrit tahmin motoru: ML, Etkinlik Etkisi ve Gerçek Zamanlı Düzeltme."""
    
    def __init__(self):
        self.model = None
        self.metadata = None
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("Model başarıyla yüklendi.")
        else:
            print("Uyarı: parking_model.joblib bulunamadı. Temel tahminler kullanılacak.")
            
        if os.path.exists(META_PATH):
            self.metadata = joblib.load(META_PATH)
            print("Model metadataları yüklendi.")
        else:
            print("Uyarı: model_metadata.joblib bulunamadı.")

    def get_base_prediction(self, dt):
        """Makine öğrenmesi modelini kullanarak temel doluluk oranını tahmin eder."""
        if not self.model or not self.metadata:
            return 0.5 * CAPACITY
        
        hour = dt.hour
        day_of_week = dt.weekday()
        month = dt.month
        is_weekend = 1 if day_of_week >= 5 else 0
        week_of_year = dt.isocalendar()[1]
        
        rolling_3h = self.metadata.get("last_rolling_3h", 0.5)
        rolling_24h = self.metadata.get("last_rolling_24h", 0.5)

        features = pd.DataFrame(
            [[hour, day_of_week, month, is_weekend, week_of_year, rolling_3h, rolling_24h]], 
            columns=self.metadata["features"]
        )
        
        rate = self.model.predict(features)[0]
        return rate * CAPACITY

    def calculate_event_impact(self, target_dt, event_start, event_end, extra_cars):
        """Gaussian dağılımı kullanarak etkinliklerin otopark yoğunluğuna etkisini hesaplar."""
        start = datetime.strptime(event_start, "%Y-%m-%dT%H:%M")
        end = datetime.strptime(event_end, "%Y-%m-%dT%H:%M")
        
        center = start + (end - start) / 2
        duration_hours = (end - start).total_seconds() / 3600
        diff_hours = (target_dt - center).total_seconds() / 3600
        
        sigma = duration_hours / 4 
        if sigma == 0:
            return 0
        
        impact_factor = math.exp(-(diff_hours**2) / (2 * (sigma**2)))
        
        if abs(diff_hours) > duration_hours:
            return 0
        
        return extra_cars * impact_factor

    def predict(self, target_dt):
        """
        Tahmin Motoru Mantığı:
        1. ML Tahmini: Geçmiş verilerdeki trendleri (saat, gün vb.) analiz eder.
        2. Etkinlik Etkisi: Gaussian (çan eğrisi) dağılımı ile etkinlik yoğunluğunu hesaplar.
        3. Üssel Düzeltme: Mevcut otopark durumu ile tahmin arasındaki farkı sönümleyerek uygular.
        """
        data = load_data()
        now = datetime.now()
        
        # 1. Temel ML Tahmini
        base_pred = self.get_base_prediction(target_dt)
        reasons = []
        
        # 2. Etkinlik Etkisi (Gaussian Yoğunluk Hesabı)
        event_impact = 0
        for event in data["events"]:
            impact = self.calculate_event_impact(target_dt, event["start"], event["end"], event["extra_cars"])
            if impact > 1:
                event_impact += impact
                intensity = int((impact / event['extra_cars']) * 100)
                reasons.append(f"Etkinlik Etkisi (Yoğunluk: %{intensity})")

        prediction = base_pred + event_impact

        # 3. Gerçek Zamanlı Üssel Düzeltme (Exponential Decay)
        hours_diff = abs((target_dt - now).total_seconds()) / 3600
        k = 0.7  # Sönümleme katsayısı (zaman geçtikçe etkinin azalma hızı)
        decay = math.exp(-k * hours_diff)
        
        current_occ = data["current_occupancy"]
        expected_now = self.get_base_prediction(now)
        delta = current_occ - expected_now
        
        adj_value = delta * decay
        prediction += adj_value

        
        if abs(adj_value) > 2:
            reasons.append(f"Anlık Veri Düzeltmesi (Etki: {decay:.2f})")

        # 4. Sınırlandırma (0 - Kapasite)
        prediction = max(0, min(CAPACITY, prediction))
        occupancy_rate = (prediction / CAPACITY) * 100
        
        level = "Rahat" if occupancy_rate < 40 else "Orta" if occupancy_rate < 80 else "Yoğun"
        
        return {
            "predicted_occupancy": round(prediction, 1),
            "occupancy_percentage": round(occupancy_rate, 1),
            "available_spaces": round(CAPACITY - prediction, 1),
            "level": level,
            "reasons": reasons if reasons else ["Normal trafik seyri"],
            "performance_info": f"Model MAE: %{round(self.metadata.get('mae', 0)*100, 2)}" if self.metadata else "Model eğitilmedi"
        }

engine = PredictionEngine()

# --- Route Tanımlamaları ---

@app.route('/')
def index():
    """Kullanıcı Paneli."""
    return render_template('index.html')

@app.route('/admin')
def admin():
    """Yönetici Paneli."""
    return render_template('admin.html')

@app.route('/api/status')
def get_status():
    """Otoparkın anlık durumunu döndürür."""
    return jsonify(load_data())

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Belirli bir tarih ve saat için tahmin yapar."""
    dt_str = request.json.get("datetime")
    dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M")
    return jsonify(engine.predict(dt))

@app.route('/api/forecast')
def api_forecast():
    """Önümüzdeki 12 saat için saatlik tahmin listesi döndürür."""
    results = []
    now = datetime.now()
    for i in range(12):
        future = now + timedelta(hours=i)
        res = engine.predict(future)
        results.append({
            "time": future.strftime("%H:00"), 
            "occupancy": res["predicted_occupancy"]
        })
    return jsonify(results)

# --- Yönetici API İşlemleri ---

@app.route('/api/admin/occupancy', methods=['POST'])
def update_occupancy():
    """Manuel doluluk artırma/azaltma."""
    data = load_data()
    action = request.json.get("action")
    if action == "inc":
        data["current_occupancy"] = min(CAPACITY, data["current_occupancy"] + 1)
    else:
        data["current_occupancy"] = max(0, data["current_occupancy"] - 1)
    save_data(data)
    return jsonify(data)

@app.route('/api/admin/slot', methods=['POST'])
def toggle_slot():
    """Belirli bir park yerini rezerve etme/boşaltma."""
    data = load_data()
    idx = request.json.get("index")
    data["slots"][idx] = not data["slots"][idx]
    data["current_occupancy"] = sum(data["slots"])
    save_data(data)
    return jsonify(data)

@app.route('/api/admin/event', methods=['POST'])
def add_event():
    """Sisteme yeni bir etkinlik ekler."""
    data = load_data()
    data["events"].append(request.json)
    save_data(data)
    return jsonify(data)

@app.route('/api/admin/event/<int:index>', methods=['DELETE'])
def delete_event(index):
    """Sistemden bir etkinliği siler."""
    data = load_data()
    data["events"].pop(index)
    save_data(data)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

