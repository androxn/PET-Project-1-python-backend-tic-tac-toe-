from sklearn.model_selection import train_test_split

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_student_data
from src.modeling import (
    build_models,
    evaluate_binary_model,
    find_best_threshold,
    positive_class_probabilities,
)


def test_generate_student_data_shape_and_columns():
    data = generate_student_data(n_samples=200, random_state=42)

    assert data.shape == (200, len(FEATURE_COLUMNS) + 1)
    assert list(data.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]


def test_generate_student_data_has_both_classes():
    data = generate_student_data(n_samples=200, random_state=42)

    assert set(data[TARGET_COLUMN].unique()) == {0, 1}


def test_logistic_regression_pipeline_runs():
    data = generate_student_data(n_samples=250, random_state=42)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_models(random_state=42)["Logistic Regression"]
    model.fit(X_train, y_train)

    probabilities = positive_class_probabilities(model, X_test)
    threshold, _ = find_best_threshold(y_test, probabilities)
    metrics = evaluate_binary_model(model, X_test, y_test, threshold=threshold)

    assert 0 <= threshold <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["confusion_matrix"].shape == (2, 2)
