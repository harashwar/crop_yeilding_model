"""
╔══════════════════════════════════════════════════════════════════════╗
║   AI-Powered Crop Yield Prediction — Flask Web API                  ║
║   app.py — Backend Server with Live Weather Integration             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import joblib
import json
import os
import requests
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)

# Load model when server starts (important for Render)
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ---------------------------
# Paths
# ---------------------------
MODEL_PATH = "model/crop_yield_model.pkl"
ENCODERS_PATH = "model/label_encoders.pkl"
METADATA_PATH = "model/model_metadata.json"

model = None
label_encoders = None
metadata = None
soil_data = {}

# ---------------------------
# Load Model
# ---------------------------
def load_model():
    global model, label_encoders, metadata, soil_data

    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    with open("soil_nutrients_dataset.geojson") as f:
        geojson = json.load(f)

        for feature in geojson["features"]:
            p = feature["properties"]
            district = p["district"]

            soil_data[district] = {
                "N": float(p["N_percentage"]),
                "P": float(p["P_percentage"]),
                "K": float(p["K_percentage"]),
                "OC": float(p["OC_percentage"]),
                "pH": float(p["pH_percentage"]),
                "EC": float(p["EC_percentage"]),
                "S": float(p["S_percentage"]),
                "Fe": float(p["Fe_percentage"]),
                "Zn": float(p["Zn_percentage"]),
                "Cu": float(p["Cu_percentage"]),
                "B": float(p["B_percentage"]),
                "Mn": float(p["Mn_percentage"]),
            }

    print("Model loaded successfully")

# 🔴 IMPORTANT
load_model()

# ──────────────────────────────────────────────────────────
# Load model artifacts
# ──────────────────────────────────────────────────────────
MODEL_PATH    = "model/crop_yield_model.pkl"
ENCODERS_PATH = "model/label_encoders.pkl"
METADATA_PATH = "model/model_metadata.json"

model          = None
label_encoders = None
metadata       = None
soil_data      = {}

def load_model():
    global model, label_encoders, metadata, soil_data
    try:
        model          = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(ENCODERS_PATH)
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        
        # Load Soil Nutrients
        with open("soil_nutrients_dataset.geojson") as f:
            geojson = json.load(f)
            for feature in geojson.get("features", []):
                p = feature.get("properties", {})
                dist = p.get("district")
                if dist:
                    soil_data[dist] = {
                        "N": float(p.get("N_percentage", 0)),
                        "P": float(p.get("P_percentage", 0)),
                        "K": float(p.get("K_percentage", 0)),
                        "OC": float(p.get("OC_percentage", 0)),
                        "pH": float(p.get("pH_percentage", 0)),
                        "EC": float(p.get("EC_percentage", 0)),
                        "S": float(p.get("S_percentage", 0)),
                        "Fe": float(p.get("Fe_percentage", 0)),
                        "Zn": float(p.get("Zn_percentage", 0)),
                        "Cu": float(p.get("Cu_percentage", 0)),
                        "B": float(p.get("B_percentage", 0)),
                        "Mn": float(p.get("Mn_percentage", 0)),
                    }
                    
        print("✅ Model loaded successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Please run: python train_model.py")
        return False

# ──────────────────────────────────────────────────────────
# Live Weather API (OpenWeatherMap — free tier)
# ──────────────────────────────────────────────────────────
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"   # <-- Replace or leave for demo
WEATHER_BASE    = "https://api.openweathermap.org/data/2.5/weather"

# District → (latitude, longitude) mapping
DISTRICT_COORDS = {
    "Thanjavur"    : (10.787, 79.1378),
    "Madurai"      : (9.9252, 78.1198),
    "Coimbatore"   : (11.0168, 76.9558),
    "Salem"        : (11.6643, 78.1460),
    "Erode"        : (11.3410, 77.7172),
    "Trichy"       : (10.7905, 78.7047),
    "Dindigul"     : (10.3624, 77.9695),
    "Tirunelveli"  : (8.7139,  77.7567),
    "Kanyakumari"  : (8.0883,  77.5385),
    "Namakkal"     : (11.2189, 78.1673),
    "Karur"        : (10.9601, 78.0766),
    "Vellore"      : (12.9165, 79.1325),
    "Tiruppur"     : (11.1085, 77.3411),
    "Cuddalore"    : (11.7480, 79.7714),
    "Nagapattinam" : (10.7672, 79.8449),
    "Thoothukudi"  : (8.7642,  78.1348),
    "Krishnagiri"  : (12.5266, 78.2138),
}

def get_live_weather(district: str):
    """Fetch live temperature and humidity from OpenWeatherMap."""
    coords = DISTRICT_COORDS.get(district)
    if not coords:
        return None, "District coordinates not found."
    if WEATHER_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
        return None, "Weather API key not configured. Please enter manually."
    try:
        lat, lon = coords
        resp = requests.get(WEATHER_BASE, params={
            "lat"   : lat,
            "lon"   : lon,
            "appid" : WEATHER_API_KEY,
            "units" : "metric",
        }, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        temp     = round(data["main"]["temp"], 1)
        humidity = round(data["main"]["humidity"], 1)
        weather_desc = data["weather"][0]["description"].title()
        return {
            "temperature" : temp,
            "humidity"    : humidity,
            "description" : weather_desc,
            "city"        : data.get("name", district),
        }, None
    except requests.exceptions.RequestException as e:
        return None, f"Weather fetch failed: {str(e)}"

# ──────────────────────────────────────────────────────────
# Prediction Helper
# ──────────────────────────────────────────────────────────
FEATURE_COLS = ['district', 'N', 'P', 'K', 'OC', 'pH', 'EC', 'S', 'Fe', 'Zn', 'Cu', 'B', 'Mn', 'temperature', 'humidity', 'crop', 'Soil_Type']

def predict_yield(district, humidity, temperature, crop_type, soil_type):
    """Run prediction through the Random Forest model."""
    if model is None or label_encoders is None:
        return None, "Model not loaded."
    
    # Check if we have soil info for that district
    if district not in soil_data:
        return None, f"Soil data not found for district {district}."
        
    soil = soil_data[district]
    
    try:
        # Encode categoricals and combine with nutrients
        encoded = {
            'district'   : label_encoders['district'].transform([district])[0],
            'N'          : soil['N'],
            'P'          : soil['P'],
            'K'          : soil['K'],
            'OC'         : soil['OC'],
            'pH'         : soil['pH'],
            'EC'         : soil['EC'],
            'S'          : soil['S'],
            'Fe'         : soil['Fe'],
            'Zn'         : soil['Zn'],
            'Cu'         : soil['Cu'],
            'B'          : soil['B'],
            'Mn'         : soil['Mn'],
            'temperature': float(temperature),
            'humidity'   : float(humidity),
            'crop'       : label_encoders['crop'].transform([crop_type])[0],
            'Soil_Type'  : label_encoders['Soil_Type'].transform([soil_type])[0],
        }
        df_input = pd.DataFrame([encoded], columns=FEATURE_COLS)
        pred = float(model.predict(df_input)[0])
        return round(pred, 4), None
    except ValueError as e:
        return None, f"Invalid input value: {str(e)}"

# ──────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/model-report")
def model_report():
    """Serve the ML training report dashboard."""
    return render_template("model_report.html")

@app.route("/api/metadata", methods=["GET"])
def api_metadata():
    """Return model metadata and available options."""
    if metadata is None:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify(metadata)

@app.route("/api/weather", methods=["GET"])
def api_weather():
    """Get live weather data for a district."""
    district = request.args.get("district", "")
    if not district:
        return jsonify({"error": "District parameter required"}), 400
    weather, err = get_live_weather(district)
    if err:
        return jsonify({"error": err}), 200  # 200 so frontend can show the message
    return jsonify(weather)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict crop yield."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    required = ["district", "humidity", "temperature", "crop_type", "soil_type"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    district    = data["district"]
    humidity    = data["humidity"]
    temperature = data["temperature"]
    crop_type   = data["crop_type"]
    soil_type   = data["soil_type"]

    pred, err = predict_yield(district, humidity, temperature, crop_type, soil_type)
    if err:
        return jsonify({"error": err}), 400

    # Build confidence range (±5% based on model RMSE)
    rmse   = metadata.get("rf_rmse", 0.1)
    low    = round(max(0, pred - rmse), 3)
    high   = round(pred + rmse, 3)

    # Yield quality label
    if pred >= 2.0:
        quality = "Excellent"
        emoji   = "🌟"
        color   = "#22c55e"
    elif pred >= 1.7:
        quality = "Good"
        emoji   = "✅"
        color   = "#84cc16"
    elif pred >= 1.4:
        quality = "Average"
        emoji   = "⚠️"
        color   = "#f59e0b"
    else:
        quality = "Low"
        emoji   = "🔴"
        color   = "#ef4444"

    return jsonify({
        "predicted_yield" : pred,
        "unit"            : "Ton/Hectare",
        "confidence_low"  : low,
        "confidence_high" : high,
        "quality_label"   : quality,
        "quality_emoji"   : emoji,
        "quality_color"   : color,
        "inputs"          : {
            "district"    : district,
            "humidity"    : humidity,
            "temperature" : temperature,
            "crop_type"   : crop_type,
            "soil_type"   : soil_type,
            "soil_nutrients": soil_data.get(district, {})
        }
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    loaded = load_model()
    if not loaded:
        print("\n⚠️  Model not found. Training now...")
        import subprocess, sys
        subprocess.run([sys.executable, "train_model.py"], check=True)
        load_model()
    print("\n🌐 Starting web server at http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False)
