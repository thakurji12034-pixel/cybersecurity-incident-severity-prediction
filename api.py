from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "random_forest.joblib"
COLUMNS_FILE = BASE_DIR / "models" / "feature_columns.joblib"
METRICS_FILE = BASE_DIR / "outputs" / "model_comparison.csv"

# main.py ke jaisa hi: ye columns category (text) ki tarah treat hote hain
ID_AS_CATEGORY = ["OrgId", "DetectorId", "AlertTitle",
                  "ApplicationId", "OAuthApplicationId",
                  "ApplicationName", "OSFamily", "OSVersion",
                  "CountryCode", "State", "City"]

app = FastAPI(title="Cyber Incident Severity API", version="1.0")

# Model start hote hi ek baar load hota hai
model = None
feature_columns = None
if MODEL_FILE.exists() and COLUMNS_FILE.exists():
    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(COLUMNS_FILE)


class PredictRequest(BaseModel):
    record: Dict[str, Any]   # ek incident record, {"column": value, ...}


def to_text(v):
    """123 aur 123.0 ko same text 123 banao. Missing 'nan' ban jata hai."""
    if pd.isna(v):
        return "nan"
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(v)
    except (TypeError, ValueError):
        return str(v)


def prepare_record(record: dict) -> pd.DataFrame:
    row = pd.DataFrame([record])

    # time features (main.py jaisa)
    if "Timestamp" in row.columns:
        t = pd.to_datetime(row["Timestamp"], errors="coerce", utc=True)
        row["Hour"] = t.dt.hour
        row["DayOfWeek"] = t.dt.day_of_week
        row["Month"] = t.dt.month

    # sirf wahi columns, wahi order me, jo training me the
    X = row.reindex(columns=feature_columns)
    X = X.astype(object).where(X.notna(), np.nan)

    pre = model.named_steps["preprocessor"]
    numeric_cols = list(pre.transformers_[0][2])

    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    for c in ID_AS_CATEGORY:
        if c in X.columns and c not in numeric_cols:
            X[c] = X[c].map(to_text)

    return X


@app.get("/")
def home():
    return {"message": "Cyber Incident Severity API is running. See /docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="model_comparison.csv not found. Run main.py first.")
    return pd.read_csv(METRICS_FILE).to_dict(orient="records")


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Run main.py first.")
    try:
        X_new = prepare_record(request.record)
        prediction = model.predict(X_new)[0]
        probabilities = model.predict_proba(X_new)[0]
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not predict: {error}")

    return {
        "prediction": str(prediction),
        "probabilities": {str(c): float(p) for c, p in zip(model.classes_, probabilities)}
    }