# Student Dropout Risk Prediction

This repository contains an application project for a summer school in Machine Learning.

The goal is to build a clear and reproducible machine learning pipeline that predicts whether a student is at risk of dropping out based on academic activity, attendance, assignments, and engagement.

The project is intentionally built at a strong beginner / junior level:
it uses Python, NumPy, Pandas, Matplotlib, and scikit-learn, without advanced deep learning frameworks.

## What the project demonstrates

- Synthetic data generation with NumPy and Pandas
- Basic exploratory data analysis
- Baseline model comparison
- Logistic Regression as an interpretable model
- Random Forest as a stronger non-linear comparison model
- Cross-validation
- Classification metrics
- Threshold tuning
- Feature importance analysis
- Basic error analysis
- Markdown reporting in `SOLUTION.md`

## Repository structure

```text
.
├── main.py
├── run_checks.py
├── SOLUTION.md
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── modeling.py
│   └── visualization.py
├── data/
│   └── .gitkeep
├── reports/
│   ├── .gitkeep
│   └── figures/
│       └── .gitkeep
└── tests/
    └── test_project.py
```

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

You can also change the dataset size and random seed:

```bash
python main.py --n-samples 1000 --random-state 42
```

Run basic project checks:

```bash
python run_checks.py
```

The script will generate the dataset, train models, print metrics, and save reports and plots.

## Main models

The project compares:

- Dummy Classifier
- Logistic Regression
- Random Forest

Logistic Regression is used as the main interpretable model, while Random Forest is used as a stronger non-linear comparison model.
