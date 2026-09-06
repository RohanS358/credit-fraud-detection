<img src="logo.svg" width="72" height="72" alt="logo" />

# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)

Exploratory analysis and classification model for detecting fraudulent credit card transactions, built in a single Jupyter notebook (`KCE080BCT033.ipynb`).

## Dataset

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle) — simulated legitimate and fraudulent transactions from **1 Jan 2019 to 31 Dec 2020**, covering **1,000 customers** across a pool of **800 merchants** (555,719 rows, 33 columns before cleaning). Columns include transaction time/amount, card and merchant identifiers, cardholder demographics (name, gender, address, job, DOB), merchant/customer geo-coordinates, and the `is_fraud` label.

## What the notebook does

- **Data cleaning** — median-fills numeric nulls, `"Unknown"`-fills categorical nulls, parses date columns, casts `is_fraud` to boolean, label-encodes categorical columns.
- **Outlier handling** — IQR-based filtering on numeric columns (555,719 → 373,641 rows).
- **Exploratory analysis**:
  - Histograms of numeric features and top-category bar charts for categoricals.
  - Spatial analysis: haversine distance between cardholder and merchant location, fraud rate by city population and by state (no meaningful correlation found between distance and fraud in this dataset).
  - Time-based patterns: fraud rate by hour of day and day of week (fraud rate spikes late at night, 22:00-23:00), time between consecutive transactions per card, recurring transactions.
  - Spending behavior: fraud rate vs. transaction amount, deviation from a cardholder's average spend, merchant categories most targeted by fraud (food_dining, gas_transport, grocery_net, health_fitness, kids_pets, misc_pos, personal_care, travel).
  - Demographics: fraud rate by gender and by age group, top job titles by fraud rate.
  - Cross-location behavior: transactions far from home location, cross-state transaction counts.
- **Feature engineering** — rolling 24h/7d transaction counts and average/sum spend per card, a composite rule-based `fraud_score` combining distance, spending deviation, low-amount transactions, high-fraud categories, and late-night timing.
- **Feature reduction** — drops features with pairwise correlation > 0.95; also explores PCA (95% variance retained).
- **Scaling** — log1p transform on highly skewed numeric features (|skew| > 1.0), then `StandardScaler`.
- **Modeling** — class imbalance handled with SMOTE oversampling (original: 319 fraud vs. 298,593 legitimate in the training split); trains and compares `RandomForestClassifier`, `LogisticRegression`, and `MLPClassifier`.

## Results

| Model | Accuracy | F1 Score | R2 Score |
|---|---|---|---|
| **RandomForest** | **99.92%** | 0.37 | 0.14 |
| MLPClassifier | 99.85% | 0.30 | -0.58 |
| LogisticRegression | 82.35% | 0.01 | -182.36 |

RandomForest was selected as the best model. Note the dataset is heavily imbalanced (fraud is ~0.1% of transactions), so accuracy alone is not a reliable metric — the low F1 scores across all models reflect that imbalance even after SMOTE resampling.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook KCE080BCT033.ipynb
```

Download the dataset from Kaggle and update the file path used in `load_and_clean_data(filepath)` before running the notebook top to bottom.

## Tech stack

pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn, haversine, joblib.
