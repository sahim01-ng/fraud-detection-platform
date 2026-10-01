"""
FastAPI service that scores transactions.

Run:   uvicorn app:app --reload --port 8000
Docs:  http://localhost:8000/docs   (interactive page to test /predict)
"""
import time
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

import scoring

app = FastAPI(title="Fraud Detection ML Service", version="1.0")


class Transaction(BaseModel):
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "TRANSFER",
                "amount": 181000.0,
                "oldbalanceOrg": 181000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "hour": 2,
            }
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": scoring.MODEL_NAME}


@app.post("/predict")
def predict(tx: Transaction):
    start = time.perf_counter()
    result = scoring.score_transaction(tx.model_dump())
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result
