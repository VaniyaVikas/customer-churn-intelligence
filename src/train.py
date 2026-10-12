
import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, METRICS_PATH, RANDOM_STATE
from src.data_loader import load_data
from src.preprocessing import clean_data, split_features_target
from src.feature_engineering import build_preprocessor


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def train():
    df = clean_data(load_data())
    X, y = split_features_target(df)

    if y.nunique() != 2:
        raise ValueError("Training requires both churn classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    best_name = None
    best_score = -1
    best_pipeline = None

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics

        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Choose by ROC-AUC, not training accuracy
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_name = name
            best_pipeline = pipeline

    joblib.dump(best_pipeline, MODEL_PATH)

    output = {
        "best_model": best_name,
        "test_rows": int(len(X_test)),
        "metrics": results,
    }
    METRICS_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    train()
