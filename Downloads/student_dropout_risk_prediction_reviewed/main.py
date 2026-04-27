from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data import TARGET_COLUMN, generate_student_data
from src.modeling import (
    build_models,
    cross_validation_report,
    evaluate_binary_model,
    find_best_threshold,
    logistic_coefficients,
    positive_class_probabilities,
    random_forest_feature_importance,
)
from src.visualization import (
    save_confusion_matrix,
    save_feature_histograms,
    save_feature_importance,
    save_metric_comparison,
    save_model_curves,
    save_target_distribution,
    save_threshold_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Student dropout risk prediction project.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of synthetic samples.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def format_metric(value: float) -> str:
    if value != value:  # NaN check
        return "nan"
    return f"{value:.3f}"


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    print("-" * 50)
    print(f"Accuracy:  {format_metric(metrics['accuracy'])}")
    print(f"Precision: {format_metric(metrics['precision'])}")
    print(f"Recall:    {format_metric(metrics['recall'])}")
    print(f"F1-score:  {format_metric(metrics['f1'])}")
    print(f"ROC-AUC:   {format_metric(metrics['roc_auc'])}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


def main() -> None:
    args = parse_args()

    reports_dir = Path("reports")
    output_dir = reports_dir / "figures"
    data_path = Path("data") / "student_dropout_data.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_student_data(n_samples=args.n_samples, random_state=args.random_state)
    data.to_csv(data_path, index=False)

    print("Dataset shape:", data.shape)
    print("\nTarget distribution:")
    print(data[TARGET_COLUMN].value_counts(normalize=True).rename("share"))

    save_target_distribution(data, output_dir)
    save_feature_histograms(data, output_dir)

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=args.random_state,
        stratify=y,
    )

    models = build_models(random_state=args.random_state)

    print("\nCross-validation results:")
    cv_results = cross_validation_report(models, X_train, y_train)
    print(cv_results.round(3).to_string(index=False))
    cv_results.to_csv(reports_dir / "cross_validation_results.csv", index=False)
    save_metric_comparison(cv_results, output_dir)

    fitted_models = {}
    test_metrics = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        metrics = evaluate_binary_model(model, X_test, y_test)
        print_metrics(name, metrics)

        test_metrics.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    main_model_name = "Logistic Regression"
    main_model = fitted_models[main_model_name]

    probabilities = positive_class_probabilities(main_model, X_test)
    best_threshold, threshold_report = find_best_threshold(y_test, probabilities)
    threshold_report.to_csv(reports_dir / "threshold_report.csv", index=False)
    save_threshold_curve(threshold_report, output_dir)

    tuned_metrics = evaluate_binary_model(
        main_model,
        X_test,
        y_test,
        threshold=best_threshold,
    )

    print(f"\nBest threshold for Logistic Regression: {best_threshold:.2f}")
    print_metrics("Logistic Regression with tuned threshold", tuned_metrics)

    test_metrics.append(
        {
            "model": "Logistic Regression tuned threshold",
            "accuracy": tuned_metrics["accuracy"],
            "precision": tuned_metrics["precision"],
            "recall": tuned_metrics["recall"],
            "f1": tuned_metrics["f1"],
            "roc_auc": tuned_metrics["roc_auc"],
        }
    )

    logistic_importance = logistic_coefficients(main_model, X.columns)
    logistic_importance.to_csv(reports_dir / "logistic_regression_coefficients.csv", index=False)

    forest_model = fitted_models["Random Forest"]
    forest_importance = random_forest_feature_importance(forest_model, X.columns)
    forest_importance.to_csv(reports_dir / "random_forest_feature_importance.csv", index=False)

    print("\nTop Logistic Regression coefficients:")
    print(logistic_importance.head(10).round(3).to_string(index=False))

    print("\nTop Random Forest feature importance:")
    print(forest_importance.head(10).round(3).to_string(index=False))

    save_model_curves(main_model, X_test, y_test, output_dir, main_model_name)
    save_confusion_matrix(tuned_metrics["confusion_matrix"], output_dir)
    save_feature_importance(
        logistic_importance,
        output_dir,
        "feature_importance_logistic_regression.png",
    )
    save_feature_importance(
        forest_importance,
        output_dir,
        "feature_importance_random_forest.png",
    )

    import pandas as pd

    pd.DataFrame(test_metrics).to_csv(reports_dir / "test_metrics.csv", index=False)

    print("\nSaved generated data, reports, and figures.")


if __name__ == "__main__":
    main()
