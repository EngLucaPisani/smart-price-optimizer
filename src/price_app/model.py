from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib

# Root del progetto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dove salviamo i modelli
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "model.pkl"

# Dove salviamo i CSV caricati
DATA_DIR = PROJECT_ROOT / "data" / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Colonne richieste (case-insensitive)
REQUIRED_COLS = ["company", "product_model", "feature", "price"]

# Sinonimi -> colonne standard (accetta anche italiano)
COLMAP = {
    "azienda": "company",
    "nome azienda": "company",
    "brand": "company",

    "modello": "product_model",
    "modello prodotto": "product_model",

    "caratteristica": "feature",
    "caratteristiche": "feature",
    "variant": "feature",

    "prezzo": "price",
    "costo": "price",
    "price (€)": "price",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {c: COLMAP.get(c, c) for c in df.columns}
    return df.rename(columns=rename_map)

def _build_pipeline() -> Pipeline:
    cat_features = ["company", "product_model", "feature"]
    pre = ColumnTransformer(
        transformers=[("cats", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("linreg", LinearRegression())])

def train_from_csv(csv_path: Path) -> Pipeline:
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Expected {REQUIRED_COLS}")

    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    X = df[["company", "product_model", "feature"]]
    y = df["price"]

    pipe = _build_pipeline()
    pipe.fit(X, y)

    joblib.dump(pipe, MODEL_PATH)
    return pipe

def load_model() -> Pipeline | None:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

# Demo opzionale: crea un modello fittizio se esegui direttamente il modulo
def train_and_save_dummy():
    demo = pd.DataFrame({
        "company": ["A","A","B","B","C","C"],
        "product_model": ["M1","M2","M1","M3","M2","M3"],
        "feature": ["base","pro","base","max","pro","max"],
        "price": [100,130,110,160,140,170]
    })
    pipe = _build_pipeline()
    pipe.fit(demo[["company","product_model","feature"]], demo["price"])
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved dummy model to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_dummy()
