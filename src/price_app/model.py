from pathlib import Path
import sys, traceback
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# Project root: .../smart-price-optimizer
ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

print(f"[INFO] ROOT       = {ROOT}")
print(f"[INFO] DATA_CSV   = {DATA_CSV}")
print(f"[INFO] MODEL_PATH = {MODEL_PATH}")

def train():
    try:
        if not DATA_CSV.exists():
            print(f"[ERROR] Missing dataset: {DATA_CSV}")
            print("[HINT] Create data/train.csv with columns: brand,category,size,base_cost,price")
            sys.exit(1)

        df = pd.read_csv(DATA_CSV)
        print(f"[INFO] df.shape   = {df.shape}")
        print(f"[INFO] df.columns = {list(df.columns)}")
        if df.empty:
            print("[ERROR] train.csv is empty")
            sys.exit(1)

        required = {"brand","category","size","base_cost","price"}
        missing = required - set(df.columns)
        if missing:
            print(f"[ERROR] Missing columns in CSV: {missing}")
            sys.exit(1)

        X = df[["brand", "category", "size", "base_cost"]]
        y = df["price"]

        # Train/test split (just to verify metrics)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"[INFO] X_train={X_train.shape}, X_test={X_test.shape}")

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["brand", "category"]),
                ("num", "passthrough", ["size", "base_cost"]),
            ]
        )

        pipe = Pipeline(steps=[
            ("pre", pre),
            ("model", LinearRegression())
        ])

        pipe.fit(X_train, y_train)
        preds_train = pipe.predict(X_train)
        preds_test  = pipe.predict(X_test)

        r2_tr  = r2_score(y_train, preds_train)
        mae_tr = mean_absolute_error(y_train, preds_train)
        r2_te  = r2_score(y_test, preds_test)
        mae_te = mean_absolute_error(y_test, preds_test)

        print(f"[METRICS] TRAIN  R2={r2_tr:.3f} | MAE={mae_tr:.2f}")
        print(f"[METRICS] TEST   R2={r2_te:.3f} | MAE={mae_te:.2f}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        print(f"[OK] Saved model → {MODEL_PATH}")
        print(f"[CHECK] Exists? {MODEL_PATH.exists()} | Size: {MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 'N/A'} bytes")

    except Exception as e:
        print("[EXCEPTION] Training failed:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    train()
