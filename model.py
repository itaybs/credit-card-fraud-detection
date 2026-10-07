"""Model training, threshold tuning and evaluation.

Strategy: Logistic Regression with class_weight='balanced', followed by
decision-threshold tuning via stratified cross-validation (out-of-fold
probabilities on the training set) to maximize F1.

Run directly to train and save the model:  python model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    load_data,
    split_data,
)

ARTIFACT_PATH = Path(__file__).resolve().parent / "fraud_model.joblib"
CV_FOLDS = 5


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def tune_threshold(pipeline: Pipeline, X_train, y_train) -> dict:
    """Pick the threshold that maximizes F1 on out-of-fold CV probabilities."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof_proba)
    # precision/recall have one more element than thresholds
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[best]),
        "cv_f1": float(f1[best]),
        "cv_precision": float(precision[best]),
        "cv_recall": float(recall[best]),
        "cv_f1_at_05": float(f1_score(y_train, (oof_proba >= 0.5).astype(int))),
    }


def evaluate(y_true, proba, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm / cm.sum(axis=1, keepdims=True),
        "roc_curve": (fpr, tpr),
        "pr_curve": (prec, rec),
    }


def coefficients(pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    names = [n.split("__", 1)[1] for n in names]
    return (
        pd.DataFrame({"feature": names, "coefficient": coefs})
        .sort_values("coefficient", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


def train_and_evaluate() -> dict:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline()
    tuning = tune_threshold(pipeline, X_train, y_train)

    pipeline.fit(X_train, y_train)
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "pipeline": pipeline,
        "tuning": tuning,
        "metrics": evaluate(y_test, test_proba, tuning["threshold"]),
        "metrics_default": evaluate(y_test, test_proba, 0.5),
        "coefficients": coefficients(pipeline),
        "test_proba": test_proba,
        "y_test": y_test.to_numpy(),
        "X_test": X_test,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_test_fraud": int(y_test.sum()),
        "sklearn_version": sklearn.__version__,
    }


def save_artifacts(artifacts: dict, path: Path = ARTIFACT_PATH) -> None:
    joblib.dump(artifacts, path)


def load_artifacts(path: Path = ARTIFACT_PATH) -> dict:
    """Load saved artifacts; raise if they were built with another scikit-learn version."""
    artifacts = joblib.load(path)
    if artifacts.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Model artifact was saved with a different scikit-learn version")
    return artifacts


def predict_proba(pipeline: Pipeline, features: dict) -> float:
    return float(pipeline.predict_proba(pd.DataFrame([features]))[0, 1])


def explain(pipeline: Pipeline, features: dict) -> pd.DataFrame:
    """Per-feature contribution to the log-odds (coefficient x transformed value)."""
    prep = pipeline.named_steps["prep"]
    x = prep.transform(pd.DataFrame([features]))
    x = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    contrib = pd.DataFrame(
        {"feature": names, "contribution": x * pipeline.named_steps["clf"].coef_[0]}
    )
    return (
        contrib[contrib["contribution"] != 0]
        .sort_values("contribution", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


def _print_report(a: dict) -> None:
    t, m, d = a["tuning"], a["metrics"], a["metrics_default"]
    print(f"Train: {a['n_train']} rows | Test: {a['n_test']} rows ({a['n_test_fraud']} fraud)")
    print(f"\nThreshold tuning ({CV_FOLDS}-fold stratified CV on train):")
    print(f"  best threshold = {t['threshold']:.4f}  CV F1 = {t['cv_f1']:.4f} "
          f"(P={t['cv_precision']:.4f}, R={t['cv_recall']:.4f}) | CV F1 @0.5 = {t['cv_f1_at_05']:.4f}")
    print(f"\n{'Test metric':<12}{'tuned':>10}{'@0.5':>10}")
    for k in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
        print(f"{k:<12}{m[k]:>10.4f}{d[k]:>10.4f}")
    print("\nConfusion matrix (tuned threshold) [rows=actual 0/1, cols=pred 0/1]:")
    print(m["confusion_matrix"])
    print(np.round(m["confusion_matrix_norm"], 4))
    print("\nCoefficients:")
    print(a["coefficients"].round(4).to_string(index=False))


if __name__ == "__main__":
    artifacts = train_and_evaluate()
    save_artifacts(artifacts)
    _print_report(artifacts)
    print(f"\nSaved to {ARTIFACT_PATH}")
