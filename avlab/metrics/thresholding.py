from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class RocResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float


def roc(y_true: np.ndarray, y_score: np.ndarray) -> RocResult:
    fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=1)
    auc = roc_auc_score(y_true, y_score)
    return RocResult(fpr=fpr, tpr=tpr, thresholds=thr, auc=auc)


def threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    """
    Returns score threshold tau such that FPR <= target_fpr (closest achievable).
    We predict 1 (malware) when score >= tau.
    """
    rr = roc(y_true, y_score)
    # indices where fpr <= target
    ok = np.where(rr.fpr <= target_fpr)[0]
    if ok.size == 0:
        # Cant reach target, return highest threshold
        return float(np.max(rr.thresholds))
    # Choose threshold at the largest fpr <= target
    idx = ok[-1]
    return float(rr.thresholds[idx])


def report_at_threshold(y_true: np.ndarray, y_score: np.ndarray, tau: float) -> dict[str, Any]:
    y_pred = (y_score >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0, pos_label=1
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "tau": float(tau),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
