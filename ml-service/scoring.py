"""Loads the trained model and turns a transaction into a risk score + level."""
import os

import joblib
import pandas as pd

from features import build_features

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/fraud_model.joblib")

_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
MODEL_NAME = _bundle["model_name"]
LOW_MAX = _bundle["thresholds"]["low_max"]
HIGH_MIN = _bundle["thresholds"]["high_min"]

# Training used all CPU cores; for single-transaction scoring that only adds
# thread start-up overhead, so run inference on one core (much faster per request).
if hasattr(MODEL, "n_jobs"):
    MODEL.n_jobs = 1


def risk_level(score: float) -> str:
    if score < LOW_MAX:
        return "LOW"
    if score < HIGH_MIN:
        return "MEDIUM"
    return "HIGH"


def score_transaction(tx: dict) -> dict:
    """tx keys: type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest, hour"""
    df = pd.DataFrame([tx])
    features = build_features(df)
    score = float(MODEL.predict_proba(features)[0, 1])
    return {
        "risk_score": round(score, 4),
        "risk_level": risk_level(score),
        "model": MODEL_NAME,
    }
