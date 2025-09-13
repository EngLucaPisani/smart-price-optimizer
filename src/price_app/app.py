from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Paths
BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / "models" / "model.pkl"

# Lazy-loaded model
_model = None
def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Train it first with: python src/price_app/model.py"
            )
        print(f"[INFO] Loading model: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

# Simple choices for the demo UI
BRANDS = ["alpha", "beta", "gamma"]
CATEGORIES = ["shoes", "bag"]

@app.get("/health")
def health():
    return "OK", 200

@app.get("/")
def index():
    return render_template("index.html", brands=BRANDS, categories=CATEGORIES)

@app.post("/predict")
def predict():
    # Read form inputs
    brand = request.form.get("brand")
    category = request.form.get("category")
    size = int(request.form.get("size") or 0)
    base_cost = float(request.form.get("base_cost") or 0.0)

    # Build a 2D table (DataFrame) with the exact training columns
    row = {
        "brand": brand,
        "category": category,
        "size": size,
        "base_cost": base_cost,
    }
    X = pd.DataFrame([row], columns=["brand", "category", "size", "base_cost"])

    # Predict
    model = get_model()
    y_hat = float(model.predict(X)[0])
    suggested = np.round(y_hat, 2)

    return render_template(
        "result.html",
        brand=brand,
        category=category,
        size=size,
        base_cost=base_cost,
        predicted=suggested,
    )

@app.post("/api/predict")
def api_predict():
    data = request.get_json(force=True) or {}
    row = {
        "brand": data.get("brand"),
        "category": data.get("category"),
        "size": int(data.get("size") or 0),
        "base_cost": float(data.get("base_cost") or 0.0),
    }
    X = pd.DataFrame([row], columns=["brand", "category", "size", "base_cost"])

    model = get_model()
    y_hat = float(model.predict(X)[0])
    return jsonify({"predicted_price": round(y_hat, 2)})

if __name__ == "__main__":
    print("[BOOT] Import OK, about to run Flask…")
    print("Starting Flask on http://0.0.0.0:8000")
    app.run(debug=True, host="0.0.0.0", port=8000)
