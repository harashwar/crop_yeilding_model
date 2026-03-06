from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import requests
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
warnings.filterwarnings("ignore")


app = Flask(__name__)
CORS(app)
# -------------------------
# File paths
# -------------------------
MODEL_PATH = "model/crop_yield_model.pkl"
ENCODERS_PATH = "model/label_encoders.pkl"
METADATA_PATH = "model/model_metadata.json"

model = None
label_encoders = None
metadata = None
soil_data = {}

# -------------------------
# Load model
# -------------------------
def load_model():
    global model, label_encoders, metadata, soil_data

    try:
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
                    "Mn": float(p["Mn_percentage"])
                }

        print("Model loaded successfully")

    except Exception as e:
        print("Model loading failed:", e)


# Load model when server starts
load_model()

# -------------------------
# Prediction
# -------------------------
FEATURE_COLS = [
    'district','N','P','K','OC','pH','EC','S','Fe','Zn','Cu','B','Mn',
    'temperature','humidity','crop','Soil_Type'
]

def predict_yield(district, humidity, temperature, crop_type, soil_type):

    if model is None:
        return None, "Model not loaded"

    if district not in soil_data:
        return None, "District soil data not found"

    soil = soil_data[district]

    try:
        encoded = {
            'district': label_encoders['district'].transform([district])[0],
            'N': soil['N'],
            'P': soil['P'],
            'K': soil['K'],
            'OC': soil['OC'],
            'pH': soil['pH'],
            'EC': soil['EC'],
            'S': soil['S'],
            'Fe': soil['Fe'],
            'Zn': soil['Zn'],
            'Cu': soil['Cu'],
            'B': soil['B'],
            'Mn': soil['Mn'],
            'temperature': float(temperature),
            'humidity': float(humidity),
            'crop': label_encoders['crop'].transform([crop_type])[0],
            'Soil_Type': label_encoders['Soil_Type'].transform([soil_type])[0]
        }

        df = pd.DataFrame([encoded], columns=FEATURE_COLS)

        prediction = float(model.predict(df)[0])

        return round(prediction,4), None

    except Exception as e:
        return None, str(e)


# -------------------------
# API Routes
# -------------------------

@app.route("/")
def home():
    return jsonify({
        "message":"Crop Yield Prediction API",
        "status":"running"
    })


@app.route("/api/health")
def health():
    return jsonify({
        "status":"ok",
        "model_loaded": model is not None
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.get_json()

    required = ["district","humidity","temperature","crop_type","soil_type"]

    for r in required:
        if r not in data:
            return jsonify({"error":f"Missing field {r}"}),400

    prediction, error = predict_yield(
        data["district"],
        data["humidity"],
        data["temperature"],
        data["crop_type"],
        data["soil_type"]
    )

    if error:
        return jsonify({"error":error}),400

    return jsonify({
        "predicted_yield":prediction,
        "unit":"Ton/Hectare"
    })


# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)