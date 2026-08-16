import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


def binary_metrics(y, score, pred=None, threshold=0.5):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int) if pred is None else np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": roc_auc_score(y, score) if both else np.nan,
        "pr_auc": average_precision_score(y, score) if both else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
    }


def patient_folds(groups, n_splits=5):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    n_splits = min(int(n_splits), len(unique))
    if n_splits < 2:
        raise ValueError("at least two patients are required")
    dummy = np.zeros(len(groups))
    yield from GroupKFold(n_splits=n_splits).split(dummy, groups=groups)


def _score(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x).astype(float)


def patient_cv(model, x, y, groups, n_splits=5):
    x = np.asarray(x)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    scores = np.full(len(y), np.nan, dtype=float)
    predictions = np.full(len(y), -1, dtype=int)
    folds = []

    for fold, (train, test) in enumerate(patient_folds(groups, n_splits)):
        fitted = clone(model)
        fitted.fit(x[train], y[train])
        score = _score(fitted, x[test])
        pred = fitted.predict(x[test]).astype(int)
        scores[test] = score
        predictions[test] = pred
        row = {
            "fold": fold,
            "train": len(train),
            "test": len(test),
            "patients": len(np.unique(groups[test])),
        }
        row.update(binary_metrics(y[test], score, pred))
        folds.append(row)

    return {
        "overall": binary_metrics(y, scores, predictions),
        "folds": folds,
        "scores": scores,
        "predictions": predictions,
    }
