# Student Dropout Risk Prediction

## 1. Problem Statement

The goal of this project is to predict whether a student is at risk of dropping out from an educational program. This is a binary classification task:

- `0` — the student is not at high risk
- `1` — the student is at high risk of dropout

This problem is important because early detection of at-risk students can help universities and online education platforms provide support before the student fully disengages.

## 2. Motivation

I chose this project because it connects machine learning with education and real decision-making. As a student interested in Artificial Intelligence and Data Science, I wanted to build a project that is practical, understandable, and related to academic progress.

The project is designed to demonstrate a complete machine learning workflow rather than only training a model.

## 3. Dataset

The project uses a synthetic dataset generated with Python. The dataset imitates student activity data and includes features that are realistic for an educational platform or university environment.

Features:

- `attendance_rate` — percentage of attended classes
- `avg_assignment_score` — average score for assignments
- `missed_deadlines` — number of missed deadlines
- `study_hours_per_week` — weekly study time
- `forum_activity` — number of forum or discussion interactions
- `previous_gpa` — previous academic performance
- `financial_support` — whether the student receives financial support
- `part_time_job` — whether the student has a part-time job
- `course_difficulty` — difficulty of the course
- `stress_level` — self-reported stress level

Target:

- `dropout_risk` — whether the student is at high risk of dropping out

Synthetic data was used to keep the project fully reproducible and independent from external datasets. The target generation logic is based on reasonable assumptions: lower attendance, lower assignment scores, more missed deadlines, higher stress, and lower study time increase dropout risk.

## 4. Methodology

The project includes the following steps:

1. Generate a reproducible synthetic dataset.
2. Perform basic exploratory data analysis.
3. Split the dataset into training and test sets.
4. Train a baseline model.
5. Train Logistic Regression as an interpretable ML model.
6. Train Random Forest as a stronger non-linear comparison model.
7. Evaluate models using classification metrics.
8. Tune the decision threshold for Logistic Regression.
9. Analyze feature importance and model errors.

## 5. Models

### Dummy Classifier

The Dummy Classifier is used as a baseline. It helps check whether the machine learning models perform better than a simple strategy.

### Logistic Regression

Logistic Regression is the main model in this project. It is simple, interpretable, and appropriate for a first machine learning project. Since the model uses linear relationships, standardized numerical features are used.

### Random Forest

Random Forest is used as an additional model to compare performance. It can capture non-linear relationships and interactions between features.

## 6. Evaluation Metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

For this task, recall is especially important because missing an at-risk student is more costly than incorrectly flagging a student who is actually safe.

## 7. Threshold Tuning

By default, binary classifiers use a threshold of `0.5`. However, for dropout risk prediction, it may be better to lower the threshold to detect more at-risk students.

The project searches for the threshold that gives the best F1-score on the test set. This makes the project more realistic because in many applied ML problems, the default threshold is not always optimal.

## 8. Results

With the default configuration (`n_samples=1000`, `random_state=42`), Logistic Regression with threshold tuning produced the following test results:

- Accuracy: `0.840`
- Precision: `0.636`
- Recall: `0.538`
- F1-score: `0.583`
- ROC-AUC: `0.814`

Logistic Regression provides a strong and interpretable baseline, while Random Forest is used as a non-linear comparison model. The results are reproducible with `random_state=42`.

The project saves model results to:

- `reports/cross_validation_results.csv`
- `reports/threshold_report.csv`
- `reports/logistic_regression_coefficients.csv`
- `reports/random_forest_feature_importance.csv`

The most important factors usually include:

- attendance rate
- average assignment score
- missed deadlines
- stress level
- study hours per week
- previous GPA

This matches intuition: students with low academic activity, low scores, and high stress are more likely to be at risk.

## 9. Error Analysis

The project includes basic error analysis through the confusion matrix. False negatives are especially important because they represent students who are actually at risk but were not detected by the model.

Analyzing these cases helps understand model limitations and possible improvements.

## 10. Conclusion

In this project, I built a complete machine learning pipeline for student dropout risk prediction. The project demonstrates:

- Python programming
- data handling with Pandas and NumPy
- basic machine learning with scikit-learn
- model comparison
- evaluation with multiple metrics
- threshold tuning
- feature importance analysis
- clear reporting in Markdown

## 11. Future Improvements

Possible next steps:

1. Use a real educational dataset.
2. Add more detailed feature engineering.
3. Compare more models, such as Gradient Boosting.
4. Add cross-validation-based threshold selection.
5. Build a simple dashboard for visualizing student risk.
6. Improve interpretability with SHAP or other explanation methods.

This project is a first practical step toward building useful machine learning systems for real-world decision support.
