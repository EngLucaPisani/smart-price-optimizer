from pathlib import Path
from datetime import datetime
from io import BytesIO, StringIO
import csv
import pandas as pd

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, jsonify
)
from werkzeug.utils import secure_filename
import joblib

from .model import (
    train_from_csv, load_model, MODEL_PATH,
    DATA_DIR, REQUIRED_COLS
)

HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.secret_key = "dev-secret-change-me"  # cambia in produzione

# Dev helpers
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

# Carica il modello se presente
MODEL = load_model()

# ----------------------
#        ROUTES
# ----------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/", methods=["GET"])
def index():
    has_model = MODEL_PATH.exists()
    return render_template("index.html", has_model=has_model)

@app.route("/download-sample", methods=["GET"])
def download_sample():
    """CSV di esempio da compilare."""
    sample_rows = [
        ["company", "product_model", "feature", "price"],
        ["BrandA", "Alpha", "base", "99.9"],
        ["BrandA", "Alpha", "pro",  "129.0"],
        ["BrandB", "Beta",  "base", "109.0"],
        ["BrandB", "Beta",  "max",  "159.0"],
        ["BrandC", "Gamma", "pro",  "139.0"],
    ]
    sio = StringIO()
    csv.writer(sio).writerows(sample_rows)
    bio = BytesIO(sio.getvalue().encode("utf-8"))
    bio.seek(0)
    return send_file(bio, mimetype="text/csv", as_attachment=True,
                     download_name="sample_competitors.csv")

ALLOWED_EXTENSIONS = {"csv"}
def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html", required_cols=REQUIRED_COLS)

    # POST
    if "file" not in request.files:
        flash("No file part in request", "error")
        return redirect(url_for("upload"))
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for("upload"))
    if not allowed_file(file.filename):
        flash("Please upload a .csv file", "error")
        return redirect(url_for("upload"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = DATA_DIR / f"{ts}__{secure_filename(file.filename)}"
    file.save(dest)

    try:
        global MODEL
        MODEL = train_from_csv(dest)
        flash(f"Model trained from {dest.name}", "success")
        return redirect(url_for("predict_form"))
    except Exception as e:
        flash(f"Training failed: {e}", "error")
        return redirect(url_for("upload"))

@app.route("/predict-form", methods=["GET"])
def predict_form():
    if not MODEL_PATH.exists():
        flash("No trained model found. Upload a CSV first.", "error")
        return redirect(url_for("upload"))
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_PATH.exists():
        flash("No trained model found. Upload a CSV first.", "error")
        return redirect(url_for("upload"))

    company       = (request.form.get("company") or "").strip()
    product_model = (request.form.get("product_model") or "").strip()
    feature       = (request.form.get("feature") or "").strip()

    if not (company and product_model and feature):
        flash("Please fill all fields (company, product model, feature).", "error")
        return redirect(url_for("predict_form"))

    try:
        model = joblib.load(MODEL_PATH)

        # ✅ Usa DataFrame invece di lista di dict
        X = pd.DataFrame(
            [[company, product_model, feature]],
            columns=["company", "product_model", "feature"]
        )

        pred = model.predict(X)[0]

        return render_template(
            "result.html",
            pred_value=round(float(pred), 4),
            company=company,
            product_model=product_model,
            feature=feature,
        )
    except Exception as e:
        flash(f"Prediction failed: {e}", "error")
        return redirect(url_for("predict_form"))

# (facoltativo) API JSON
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not MODEL_PATH.exists():
        return jsonify({"error": "model not trained"}), 400
    payload = request.get_json(silent=True) or {}
    try:
        company       = (payload.get("company") or "").strip()
        product_model = (payload.get("product_model") or "").strip()
        feature       = (payload.get("feature") or "").strip()
        if not (company and product_model and feature):
            return jsonify({"error": "missing fields"}), 400
        model = joblib.load(MODEL_PATH)
        X = pd.DataFrame([[company, product_model, feature]],
                         columns=["company", "product_model", "feature"])
        pred = model.predict(X)[0]
        return jsonify({"predicted_price": round(float(pred), 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
