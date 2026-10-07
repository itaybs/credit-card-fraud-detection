# Credit Card Fraud Detection & Interactive ML Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning project that detects fraudulent credit card transactions with a **class-weighted Logistic Regression** model and a **cross-validated decision threshold**. The results are presented in an interactive **Streamlit** dashboard for model analysis and real-time risk scoring.

---

## Table of Contents

- [Overview & Business Problem](#overview--business-problem)
- [Dataset](#dataset)
- [Methodology & ML Pipeline](#methodology--ml-pipeline)
- [Model Performance & Results](#model-performance--results)
- [Dashboard Features](#dashboard-features)
- [Project Structure](#project-structure)
- [Installation & Local Execution](#installation--local-execution)
- [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Overview & Business Problem

Fraud detection is a classic **highly imbalanced classification** problem. In this 10,000-transaction dataset only **151 transactions (1.51%)** are fraudulent, a ratio of roughly **1 : 65**. A model that blindly labels every transaction as legitimate would reach **98.5% accuracy** while catching **zero fraud**, so accuracy is not a useful metric here.

Every fraud detection system must balance two kinds of error:

| Error | Meaning | Business cost |
|---|---|---|
| **False Negative** | Fraud passes as legitimate | Direct financial loss, chargebacks, and regulatory exposure |
| **False Positive** | Legitimate transaction is blocked | Customer friction, abandoned purchases, support costs, and churn |

The project optimizes this trade-off explicitly and lets users explore it interactively through an adjustable decision threshold.

---

## Dataset

`credit_card_fraud_10k.csv`: 10,000 transactions, 8 predictive features, no missing values, no duplicates.

| Feature | Type | Fraud vs. legitimate (mean) |
|---|---|---|
| `amount` | numeric | 216 vs 175 |
| `transaction_hour` | numeric (0–23) | **3.8 vs 11.7** (fraud clusters at night) |
| `merchant_category` | categorical (5) | Weak signal |
| `foreign_transaction` | binary | **54% vs 9%** |
| `location_mismatch` | binary | **48% vs 8%** |
| `device_trust_score` | numeric (0–99) | **38 vs 62** |
| `velocity_last_24h` | numeric | 3.2 vs 2.0 |
| `cardholder_age` | numeric | No signal |
| `is_fraud` | **target** | 151 fraud / 9,849 legitimate |

`transaction_id` is dropped before training to prevent identifier leakage.

---

## Methodology & ML Pipeline

```
Raw CSV ─► Drop ID ─► Stratified 80/20 split ─► ColumnTransformer ─► LogisticRegression(class_weight='balanced')
                                                 ├─ StandardScaler  (numeric)          │
                                                 ├─ passthrough     (binary)           ▼
                                                 └─ OneHotEncoder   (merchant)   5-fold stratified CV
                                                                                  out-of-fold probabilities
                                                                                       │
                                                                                       ▼
                                                                         Threshold that maximizes F1 = 0.965
```

### 1. Model: Logistic Regression
The model is a linear, interpretable baseline. Its coefficients on standardized features show directly which signals drive fraud risk. The strongest are location mismatch, foreign transaction, low device trust, late-night hours and high velocity.

### 2. Imbalance handling: class weighting
`class_weight='balanced'` makes every misclassified fraud case cost about **65× more** in the loss function than a misclassified legitimate one. This was chosen over naive resampling for three reasons:

- **Data integrity:** no synthetic records are created. SMOTE would interpolate binary and categorical columns into impossible values such as `foreign_transaction = 0.4`.
- **No information loss:** undersampling would discard about 97% of the legitimate transactions.
- **Simplicity and reproducibility:** the method is built into scikit-learn and fully deterministic.

### 3. Decision threshold tuning
Class weighting pushes predicted probabilities upward, so the default **0.5** cut-off produces many false alarms. To correct this:

1. **5-fold stratified cross-validation** is run on the **training set only**, producing an out-of-fold fraud score for every training transaction.
2. The precision-recall curve of those scores is scanned, and the threshold that **maximizes F1** is selected: **`0.965`**. Cross-validated F1 was 0.652, compared with 0.411 at the default 0.5.
3. The final model is refit on the full training set. The **test set is used only once**, for the final evaluation.

---

## Model Performance & Results

Evaluation on the held-out test set (2,000 transactions, 30 fraud cases):

| Metric | Default threshold (0.5) | **Tuned threshold (0.965)** |
|---|:---:|:---:|
| ROC AUC | 0.993 | **0.993** |
| PR AUC (Average Precision) | 0.741 | **0.741** |
| Precision | 0.261 | **0.511** |
| Recall | 1.000 | **0.767** |
| F1-Score | 0.414 | **0.613** |

> ROC AUC and PR AUC do not depend on the threshold. They measure how well the model **ranks** transactions, and 0.993 indicates excellent separation.

### Confusion matrix

| | Default (0.5): Pred. Legit | Default (0.5): Pred. Fraud | **Tuned (0.965): Pred. Legit** | **Tuned (0.965): Pred. Fraud** |
|---|:---:|:---:|:---:|:---:|
| **Actual Legit** | 1,885 | 85 | **1,948** | **22** |
| **Actual Fraud** | 0 | 30 | **7** | **23** |

**Key takeaway:** threshold tuning cuts **false alarms by 74% (85 → 22)** and **doubles precision**, at the cost of 7 missed frauds. The right operating point depends on the relative cost of a missed fraud versus a blocked customer. The dashboard's threshold slider makes that trade-off explicit.

> With only 30 fraud cases in the test set, a single misclassification moves recall by about 3.3 points. Treat the metrics as estimates with meaningful variance.

---

## Dashboard Features

### Tab 1: Model Performance & Analysis
- **KPI cards:** Precision, Recall, F1, ROC AUC and PR AUC
- **Confusion matrix:** switchable between raw counts and normalized percentages
- **ROC curve** and **precision-recall curve**, each marking the current operating point
- **Feature coefficients chart:** which features increase or decrease fraud risk
- **Dataset statistics:** class balance and per-class feature means
- **Methodology summary** with a default vs. tuned threshold comparison
- **Sidebar threshold slider:** every metric and chart updates live

### Tab 2: Interactive Fraud Prediction
- Input controls (sliders, numeric inputs, checkboxes) for all 8 features
- **Load Fraud Sample** and **Load Legitimate Sample** buttons that load real held-out test transactions
- **Predict** button for real-time inference
- Output: **fraud risk score (%)**, a final classification, a **red HIGH RISK or green LOW RISK badge**, and a **risk gauge** showing the decision threshold

> Because of class weighting, the displayed percentage is a **risk score** that overstates the true fraud probability (base rate about 1.5%). It should be compared against the threshold, not read as a calibrated probability.

---

## Project Structure

```
credit-card-fraud-detection/
├── .streamlit/
│   └── config.toml              # Theme and server settings
├── app.py                       # Streamlit dashboard (2 tabs)
├── model.py                     # Pipeline, CV threshold tuning, evaluation, persistence
├── data_loader.py               # Data loading, feature definitions, stratified split
├── credit_card_fraud_10k.csv    # Dataset (10,000 transactions)
├── fraud_model.joblib           # Pre-trained model and evaluation artifacts
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md
```

All file paths are resolved relative to the source files (`Path(__file__).parent`), so the app runs unchanged locally and in the cloud.

---

## Installation & Local Execution

**Prerequisites:** Python 3.10+ and Git.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) retrain the model and print the full evaluation report
python model.py

# 5. Launch the dashboard
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

> The app loads `fraud_model.joblib` when it is compatible with the installed scikit-learn version. Otherwise it retrains automatically on first launch, which takes a few seconds.

---

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, then **Deploy a public app from GitHub**.
4. Select the repository, branch `main`, and main file path `app.py`.
5. Under **Advanced settings**, choose Python **3.12**.
6. Click **Deploy**. Dependencies are installed from `requirements.txt` automatically.

---

## Limitations & Future Work

- **Small fraud sample:** with 151 fraud cases in total, the test metrics have high variance. Repeated stratified cross-validation would give tighter estimates.
- **Calibration:** class weighting distorts probabilities. Adding `CalibratedClassifierCV` would produce interpretable fraud probabilities.
- **Cost-sensitive threshold:** the threshold could instead minimize expected monetary loss, for example using the transaction `amount` as the cost of a missed fraud.
- **Model comparison:** benchmark against gradient boosting (XGBoost or LightGBM) and against SMOTENC resampling.

---

## License

This project is licensed under the [MIT License](LICENSE).
