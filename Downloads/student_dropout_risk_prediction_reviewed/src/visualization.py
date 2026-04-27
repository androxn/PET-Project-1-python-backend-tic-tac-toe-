from __future__ import annotations

from pathlib import Path
import logging
import os

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def save_target_distribution(data: pd.DataFrame, output_dir: Path) -> None:
    _ensure_output_dir(output_dir)

    if "dropout_risk" not in data.columns:
        raise ValueError("Column 'dropout_risk' is required.")

    counts = data["dropout_risk"].value_counts().sort_index()

    plt.figure()
    counts.plot(kind="bar")
    plt.title("Target Distribution")
    plt.xlabel("Dropout risk")
    plt.ylabel("Number of students")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png")
    plt.close()


def save_feature_histograms(data: pd.DataFrame, output_dir: Path) -> None:
    _ensure_output_dir(output_dir)

    columns = [
        "attendance_rate",
        "avg_assignment_score",
        "missed_deadlines",
        "study_hours_per_week",
        "previous_gpa",
        "stress_level",
    ]

    for column in columns:
        if column not in data.columns:
            continue

        plt.figure()
        data[column].hist(bins=30)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"{column}_distribution.png")
        plt.close()


def save_metric_comparison(cv_results: pd.DataFrame, output_dir: Path) -> None:
    _ensure_output_dir(output_dir)

    if cv_results.empty or "model" not in cv_results.columns:
        return

    metric_columns = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]
    available_metrics = [column for column in metric_columns if column in cv_results.columns]

    for metric in available_metrics:
        plt.figure()
        cv_results.set_index("model")[metric].plot(kind="bar")
        plt.title(f"Cross-validation {metric}")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_comparison.png")
        plt.close()


def save_model_curves(model, X_test, y_test, output_dir: Path, model_name: str) -> None:
    _ensure_output_dir(output_dir)

    if len(pd.Series(y_test).unique()) < 2:
        return

    plt.figure()
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve — {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()

    plt.figure()
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"Precision-Recall Curve — {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png")
    plt.close()


def save_threshold_curve(threshold_report: pd.DataFrame, output_dir: Path) -> None:
    _ensure_output_dir(output_dir)

    required_columns = {"threshold", "precision", "recall", "f1"}
    if threshold_report.empty or not required_columns.issubset(threshold_report.columns):
        return

    plt.figure()
    threshold_report.plot(x="threshold", y=["precision", "recall", "f1"])
    plt.title("Threshold Tuning")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_tuning.png")
    plt.close()


def save_confusion_matrix(matrix, output_dir: Path) -> None:
    _ensure_output_dir(output_dir)

    plt.figure()
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
    display.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def save_feature_importance(importance: pd.DataFrame, output_dir: Path, filename: str) -> None:
    _ensure_output_dir(output_dir)

    if importance.empty:
        return

    top_features = importance.head(10).sort_values("absolute_value")

    plt.figure()
    plt.barh(top_features["feature"], top_features["absolute_value"])
    plt.title("Top Feature Importance")
    plt.xlabel("Absolute value")
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
