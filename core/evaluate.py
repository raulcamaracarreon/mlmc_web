from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score,
)

# ---- Supervisado ----

def eval_classification(y_true, y_pred) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision_w": prec,
        "recall_w": rec,
        "f1_w": f1,
        "cm": cm,
        "report": report,
    }


def eval_regression(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

# ---- No supervisado ----

def eval_clustering(X_trans, labels) -> Dict[str, float]:
    out = {}
    labels_set = set(labels)
    if len(labels_set) > 1:
        try:
            out["silhouette"] = silhouette_score(X_trans, labels)
        except Exception:
            pass
        try:
            out["davies_bouldin"] = davies_bouldin_score(X_trans, labels)
        except Exception:
            pass
    return out