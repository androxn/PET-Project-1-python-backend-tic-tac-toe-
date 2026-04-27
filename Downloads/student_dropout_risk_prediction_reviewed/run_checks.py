from __future__ import annotations

from sklearn.model_selection import train_test_split

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_student_data
from src.modeling import (
    build_models,
    evaluate_binary_model,
    find_best_threshold,
    positive_class_probabilities,
)


def main() -> None:
    data = generate_student_data(n_samples=300, random_state=7)

    assert list(data.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert data.shape == (300, len(FEATURE_COLUMNS) + 1)
    assert set(data[TARGET_COLUMN].unique()) == {0, 1}

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
        stratify=y,
    )

    models = build_models(random_state=7)
    model = models["Logistic Regression"]
    model.fit(X_train, y_train)

    probabilities = positive_class_probabilities(model, X_test)
    threshold, threshold_report = find_best_threshold(y_test, probabilities)
    metrics = evaluate_binary_model(model, X_test, y_test, threshold=threshold)

    assert 0 <= threshold <= 1
    assert not threshold_report.empty
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["confusion_matrix"].shape == (2, 2)

    print("All checks passed successfully.")


if __name__ == "__main__":
    main()
