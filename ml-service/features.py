"""
Shared feature engineering.

IMPORTANT: train_model.py AND the FastAPI service (app.py, built next) both import
this file, so the model sees exactly the same features at training time and at
prediction time. Never duplicate this logic elsewhere.
"""
import pandas as pd

TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",   # does the sender's balance change match the amount?
    "errorBalanceDest",   # does the receiver's balance change match the amount?
    "amountToBalanceRatio",  # how much of the sender's balance is being moved
    "isEmptiedAccount",   # sender balance drained to zero (classic fraud pattern)
    "hour",               # hour of day (0-23)
    "isNight",            # 00:00-05:59 transactions are riskier
] + [f"type_{t}" for t in TX_TYPES]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns required:
        type, amount, oldbalanceOrg, newbalanceOrig,
        oldbalanceDest, newbalanceDest, hour
    Returns a DataFrame with exactly FEATURE_COLUMNS (in this order).
    """
    out = pd.DataFrame(index=df.index)

    out["amount"] = df["amount"].astype(float)
    out["oldbalanceOrg"] = df["oldbalanceOrg"].astype(float)
    out["newbalanceOrig"] = df["newbalanceOrig"].astype(float)
    out["oldbalanceDest"] = df["oldbalanceDest"].astype(float)
    out["newbalanceDest"] = df["newbalanceDest"].astype(float)

    # Balance-consistency errors: legit transactions usually have ~0 error
    out["errorBalanceOrig"] = out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    out["errorBalanceDest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]

    out["amountToBalanceRatio"] = out["amount"] / (out["oldbalanceOrg"] + 1.0)
    out["isEmptiedAccount"] = (
        (out["oldbalanceOrg"] > 0) & (out["newbalanceOrig"] == 0)
    ).astype(int)

    out["hour"] = df["hour"].astype(int) % 24
    out["isNight"] = (out["hour"] < 6).astype(int)

    for t in TX_TYPES:
        out[f"type_{t}"] = (df["type"] == t).astype(int)

    return out[FEATURE_COLUMNS]
