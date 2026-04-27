from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(random_state: int = 42) -> dict[str, Any]:
    """
    Create models for comparison.
    """
    return {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=8,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
        ),
    }


def positive_class_probabilities(model: Any, X) -> np.ndarray:
    """
    Return probabilities for class 1 when possible.

    The function is defensive and also works with scikit-learn Pipelines.
    """
    if not hasattr(model, "predict_proba"):
        return np.asarray(model.predict(X), dtype=float)

    probabilities = model.predict_proba(X)
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        classes = getattr(final_step, "classes_", None)

    if probabilities.ndim != 2:
        raise ValueError("predict_proba must return a two-dimensional array.")

    if probabilities.shape[1] == 1:
        only_class = int(classes[0]) if classes is not None else 0
        return np.ones(len(X), dtype=float) if only_class == 1 else np.zeros(len(X), dtype=float)

    if classes is not None and 1 in classes:
        positive_index = list(classes).index(1)
    else:
        positive_index = 1

    return probabilities[:, positive_index].astype(float)


def evaluate_binary_model(model: Any, X_test, y_test, threshold: float = 0.5) -> dict[str, Any]:
    """
    Evaluate a binary classifier with safe metrics.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    probabilities = positive_class_probabilities(model, X_test)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
    }


def cross_validation_report(models: dict[str, Any], X_train, y_train) -> pd.DataFrame:
    """
    Run stratified cross-validation for each model.

    The number of folds is chosen safely based on the minority class size.
    """
    class_counts = pd.Series(y_train).value_counts()

    if len(class_counts) < 2:
        raise ValueError("Cross-validation requires at least two target classes.")

    n_splits = int(min(5, class_counts.min()))
    if n_splits < 2:
        raise ValueError("Each target class must contain at least two samples.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )

        row = {"model": name, "cv_folds": n_splits}
        for metric in scoring:
            row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def find_best_threshold(y_true, probabilities) -> tuple[float, pd.DataFrame]:
    """
    Search for the threshold with the best F1-score.

    The metric calculations are implemented directly with NumPy to keep the
    function fast and independent from repeated scikit-learn validation calls.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional array.")
    if len(probabilities) != len(y_true):
        raise ValueError("probabilities and y_true must have the same length.")
    if len(probabilities) == 0:
        raise ValueError("probabilities must not be empty.")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must not contain NaN or infinite values.")

    thresholds = np.arange(0.10, 0.91, 0.01)

    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        true_positive = int(((predictions == 1) & (y_true == 1)).sum())
        false_positive = int(((predictions == 1) & (y_true == 0)).sum())
        false_negative = int(((predictions == 0) & (y_true == 1)).sum())

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    report = pd.DataFrame(rows)
    best_row = report.sort_values(["f1", "recall"], ascending=False).iloc[0]

    return float(best_row["threshold"]), report


def logistic_coefficients(model: Pipeline, feature_names) -> pd.DataFrame:
    """
    Extract Logistic Regression coefficients from a pipeline.
    """
    if not hasattr(model, "named_steps") or "model" not in model.named_steps:
        raise TypeError("Expected a scikit-learn Pipeline with a final step named 'model'.")

    final_model = model.named_steps["model"]
    if not hasattr(final_model, "coef_"):
        raise TypeError("The final model does not expose coefficients.")

    coefficients = final_model.coef_[0]
    feature_names = list(feature_names)

    if len(coefficients) != len(feature_names):
        raise ValueError("Number of coefficients does not match number of feature names.")

    return _importance_frame(feature_names, coefficients, "coefficient")


def random_forest_feature_importance(model: Any, feature_names) -> pd.DataFrame:
    """
    Extract Random Forest feature importance.
    """
    if not hasattr(model, "feature_importances_"):
        raise TypeError("The model does not expose feature_importances_.")

    importances = model.feature_importances_
    feature_names = list(feature_names)

    if len(importances) != len(feature_names):
        raise ValueError("Number of importances does not match number of feature names.")

    return _importance_frame(feature_names, importances, "importance")


def _importance_frame(feature_names: list[str], values: np.ndarray, value_column: str) -> pd.DataFrame:
    absolute_values = np.abs(values)

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                value_column: values,
                "absolute_value": absolute_values,
            }
        )
        .sort_values("absolute_value", ascending=False)
        .reset_index(drop=True)
    )


def _safe_roc_auc(y_true, probabilities) -> float:
    """
    Compute ROC-AUC safely. ROC-AUC is undefined if y_true contains one class only.
    """
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, probabilities))
