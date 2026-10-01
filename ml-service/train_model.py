"""
Train and evaluate the fraud-detection model.

Usage:
    python train_model.py                      # uses data/paysim.csv if present, else synthetic data
    python train_model.py --data data/paysim.csv
    python train_model.py --synthetic          # force synthetic data (for quick testing)

Outputs (in ./artifacts):
    fraud_model.joblib        trained model (loaded by the API)
    metrics.json              precision / recall / F1 / ROC-AUC for every model
    confusion_matrix.png      for your report
    roc_curve.png             for your report
    feature_importance.png    for your report
    model_comparison.csv      table for your report
"""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from features import FEATURE_COLUMNS, build_features

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # keeps the script runnable even if xgboost is missing
    HAS_XGB = False

ARTIFACT_DIR = "artifacts"
RANDOM_STATE = 42

# Risk-level thresholds used later by the backend / API
LOW_MAX = 0.30      # score < 0.30  -> LOW
HIGH_MIN = 0.70     # score >= 0.70 -> HIGH, in between -> MEDIUM


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def load_paysim(path: str, sample_rows: int | None = None) -> pd.DataFrame:
    """PaySim columns: step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,
    nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud"""
    print(f"[data] Loading PaySim from {path} ...")
    df = pd.read_csv(path)
    if sample_rows and len(df) > sample_rows:
        # keep ALL fraud rows + a random sample of legit ones (faster training)
        fraud = df[df["isFraud"] == 1]
        legit = df[df["isFraud"] == 0].sample(sample_rows - len(fraud), random_state=RANDOM_STATE)
        df = pd.concat([fraud, legit]).sample(frac=1, random_state=RANDOM_STATE)
    df["hour"] = df["step"] % 24  # 'step' = 1 hour of simulation time
    return df[["type", "amount", "oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest", "hour", "isFraud"]].reset_index(drop=True)


def make_synthetic(n: int = 200_000, fraud_rate: float = 0.01) -> pd.DataFrame:
    """Small PaySim-like dataset so the whole pipeline can be tested without downloading anything."""
    print(f"[data] Generating {n:,} synthetic transactions (fraud rate {fraud_rate:.1%}) ...")
    rng = np.random.default_rng(RANDOM_STATE)
    types = rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"], size=n,
                       p=[0.34, 0.22, 0.35, 0.08, 0.01])
    amount = np.round(rng.lognormal(mean=9.5, sigma=1.3, size=n), 2)
    old_org = np.round(rng.lognormal(mean=11, sigma=1.5, size=n), 2)
    old_dest = np.round(rng.lognormal(mean=11, sigma=1.5, size=n), 2)
    hour = rng.integers(0, 24, size=n)

    # Legit behaviour: balances move consistently with the amount
    is_credit = types == "CASH_IN"
    new_org = np.where(is_credit, old_org + amount, np.maximum(old_org - amount, 0))
    new_dest = old_dest + amount

    is_fraud = np.zeros(n, dtype=int)
    risky = np.where(np.isin(types, ["TRANSFER", "CASH_OUT"]))[0]
    fraud_idx = rng.choice(risky, size=int(n * fraud_rate), replace=False)
    is_fraud[fraud_idx] = 1

    # Fraud pattern: drain the whole account, destination balance doesn't update properly, often at night
    amount[fraud_idx] = old_org[fraud_idx]
    new_org[fraud_idx] = 0.0
    old_dest[fraud_idx] = np.where(rng.random(len(fraud_idx)) < 0.5, 0.0, old_dest[fraud_idx])
    new_dest[fraud_idx] = np.where(rng.random(len(fraud_idx)) < 0.5, 0.0, old_dest[fraud_idx] + amount[fraud_idx])
    hour[fraud_idx] = np.where(rng.random(len(fraud_idx)) < 0.6, rng.integers(0, 6, len(fraud_idx)), hour[fraud_idx])

    return pd.DataFrame({
        "type": types, "amount": amount, "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest, "hour": hour, "isFraud": is_fraud,
    })


# ----------------------------------------------------------------------------
# 2. MODELS
# ----------------------------------------------------------------------------
def get_models(scale_pos_weight: float) -> dict:
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
    else:
        print("[warn] xgboost not installed -> using sklearn HistGradientBoosting instead. "
              "Run `pip install xgboost` to use XGBoost.")
        models["GradientBoosting(HistGB)"] = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=RANDOM_STATE)
    return models


def evaluate(name, model, X_test, y_test, threshold=0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
    }


# ----------------------------------------------------------------------------
# 3. MAIN
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/paysim.csv", help="path to the PaySim CSV")
    parser.add_argument("--synthetic", action="store_true", help="use generated data instead of PaySim")
    parser.add_argument("--sample", type=int, default=600_000,
                        help="max rows to use from PaySim (all fraud rows are always kept). 0 = use everything")
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ---- load data
    if not args.synthetic and os.path.exists(args.data):
        df = load_paysim(args.data, sample_rows=args.sample or None)
        data_source = "PaySim"
    else:
        if not args.synthetic:
            print(f"[data] {args.data} not found -> falling back to SYNTHETIC data.")
            print("       Download PaySim from Kaggle and save it as data/paysim.csv for real results.")
        df = make_synthetic()
        data_source = "synthetic"

    print(f"[data] rows={len(df):,}  fraud={df['isFraud'].sum():,} ({df['isFraud'].mean():.3%})")
    print(df["type"].value_counts().to_string(), "\n")

    # ---- features + split (stratified because the data is heavily imbalanced)
    X = build_features(df)
    y = df["isFraud"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"[train] train={len(X_train):,} test={len(X_test):,}  scale_pos_weight={scale_pos_weight:.1f}\n")

    # ---- train + evaluate every model
    results, fitted = [], {}
    for name, model in get_models(scale_pos_weight).items():
        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"[train] {name} trained in {time.time() - t0:.1f}s")
        res = evaluate(name, model, X_test, y_test)
        results.append(res)
        fitted[name] = model
        print(f"        {res}\n")

    comparison = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    comparison.to_csv(f"{ARTIFACT_DIR}/model_comparison.csv", index=False)
    print("[result] Model comparison (best first):")
    print(comparison.to_string(index=False), "\n")

    # ---- pick the best model (PR-AUC is the right metric for imbalanced fraud data)
    best_name = comparison.iloc[0]["model"]
    best = fitted[best_name]
    print(f"[result] Best model: {best_name}\n")

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(classification_report(y_test, pred, target_names=["legit", "fraud"], digits=4))

    # ---- plots for the report
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout(); plt.savefig(f"{ARTIFACT_DIR}/confusion_matrix.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    for name, m in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{ARTIFACT_DIR}/roc_curve.png", dpi=150); plt.close()

    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=FEATURE_COLUMNS).sort_values()
        plt.figure(figsize=(7, 6)); imp.plot(kind="barh")
        plt.title(f"Feature Importance - {best_name}")
        plt.tight_layout(); plt.savefig(f"{ARTIFACT_DIR}/feature_importance.png", dpi=150); plt.close()

    # ---- risk-level distribution on the test set (shows how LOW/MEDIUM/HIGH behave)
    levels = pd.cut(proba, bins=[-0.01, LOW_MAX, HIGH_MIN, 1.01], labels=["LOW", "MEDIUM", "HIGH"])
    risk_table = pd.crosstab(levels, y_test, rownames=["risk_level"], colnames=["actual_fraud"])
    print("\n[result] Risk level vs actual fraud:\n", risk_table, "\n")

    # ---- inference latency (this is your 'real-time' number for the viva)
    sample = X_test.iloc[:200]
    t0 = time.perf_counter()
    for i in range(len(sample)):
        best.predict_proba(sample.iloc[[i]])
    latency_ms = (time.perf_counter() - t0) / len(sample) * 1000
    print(f"[result] Avg single-transaction scoring latency: {latency_ms:.2f} ms")

    # ---- save everything
    joblib.dump({
        "model": best,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "thresholds": {"low_max": LOW_MAX, "high_min": HIGH_MIN},
    }, f"{ARTIFACT_DIR}/fraud_model.joblib")

    with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
        json.dump({
            "data_source": data_source,
            "rows": int(len(df)),
            "fraud_rows": int(df["isFraud"].sum()),
            "best_model": best_name,
            "models": results,
            "scoring_latency_ms": round(latency_ms, 3),
            "thresholds": {"low_max": LOW_MAX, "high_min": HIGH_MIN},
        }, f, indent=2)

    print(f"\n[done] Saved model + report files to ./{ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
