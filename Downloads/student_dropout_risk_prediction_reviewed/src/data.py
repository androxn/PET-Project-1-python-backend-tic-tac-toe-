from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "attendance_rate",
    "avg_assignment_score",
    "missed_deadlines",
    "study_hours_per_week",
    "forum_activity",
    "previous_gpa",
    "financial_support",
    "part_time_job",
    "course_difficulty",
    "stress_level",
]

TARGET_COLUMN = "dropout_risk"


def generate_student_data(n_samples: int = 1000, random_state: int | None = 42) -> pd.DataFrame:
    """
    Generate a reproducible synthetic dataset for student dropout risk prediction.

    Parameters
    ----------
    n_samples:
        Number of rows to generate. Must be at least 80, because the project uses
        stratified train/test split and cross-validation.
    random_state:
        Seed for reproducibility. Can be an integer or None.

    Returns
    -------
    pd.DataFrame
        Dataset with student features and the binary target column `dropout_risk`.
    """
    if not isinstance(n_samples, Integral):
        raise TypeError("n_samples must be an integer.")
    n_samples = int(n_samples)

    if n_samples < 80:
        raise ValueError("n_samples must be at least 80 for a stable ML experiment.")

    if random_state is not None and not isinstance(random_state, Integral):
        raise TypeError("random_state must be an integer or None.")

    rng = np.random.default_rng(None if random_state is None else int(random_state))

    attendance_rate = rng.normal(78, 15, n_samples).clip(20, 100)
    avg_assignment_score = rng.normal(72, 14, n_samples).clip(20, 100)
    missed_deadlines = rng.poisson(2.2, n_samples).clip(0, 12)
    study_hours_per_week = rng.normal(12, 5, n_samples).clip(1, 35)
    forum_activity = rng.poisson(4, n_samples).clip(0, 25)
    previous_gpa = rng.normal(3.4, 0.45, n_samples).clip(2.0, 5.0)
    financial_support = rng.integers(0, 2, n_samples)
    part_time_job = rng.integers(0, 2, n_samples)
    course_difficulty = rng.integers(1, 6, n_samples)
    stress_level = rng.normal(5.5, 2.0, n_samples).clip(1, 10)

    logits = (
        -0.055 * attendance_rate
        -0.045 * avg_assignment_score
        + 0.42 * missed_deadlines
        -0.08 * study_hours_per_week
        -0.05 * forum_activity
        -0.85 * previous_gpa
        -0.35 * financial_support
        + 0.35 * part_time_job
        + 0.28 * course_difficulty
        + 0.33 * stress_level
        + 6.2
    )

    probability = 1 / (1 + np.exp(-logits))
    dropout_risk = rng.binomial(1, probability).astype(int)

    dropout_risk = _ensure_stable_binary_target(dropout_risk, probability)

    data = pd.DataFrame(
        {
            "attendance_rate": attendance_rate,
            "avg_assignment_score": avg_assignment_score,
            "missed_deadlines": missed_deadlines,
            "study_hours_per_week": study_hours_per_week,
            "forum_activity": forum_activity,
            "previous_gpa": previous_gpa,
            "financial_support": financial_support,
            "part_time_job": part_time_job,
            "course_difficulty": course_difficulty,
            "stress_level": stress_level,
            "dropout_risk": dropout_risk,
        }
    )

    return data[FEATURE_COLUMNS + [TARGET_COLUMN]]


def _ensure_stable_binary_target(dropout_risk: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """
    Ensure that both classes are present with enough examples.

    This protects the project from rare random seeds that can produce too few
    positive or negative samples for stratified splitting and cross-validation.
    """
    n_samples = len(dropout_risk)
    minimum_class_size = max(10, int(0.08 * n_samples))
    class_counts = np.bincount(dropout_risk, minlength=2)

    if class_counts.min() >= minimum_class_size:
        return dropout_risk

    # If the random draw is too imbalanced, use probability ranking.
    # The highest-risk students become class 1; the rest become class 0.
    positive_count = max(minimum_class_size, int(0.25 * n_samples))
    positive_count = min(positive_count, n_samples - minimum_class_size)

    adjusted_target = np.zeros(n_samples, dtype=int)
    highest_risk_indices = np.argsort(probability)[-positive_count:]
    adjusted_target[highest_risk_indices] = 1

    return adjusted_target
