from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.utils import resample
import numpy as np

OUTPUT_DIR = Path("results/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_model(train_x, train_y):
    model = RandomForestClassifier(random_state=1, max_depth=5, n_estimators=10, min_samples_leaf=50,)
    cv = TimeSeriesSplit(n_splits=5)

    scores = cross_validate(
        model, train_x, train_y, cv=cv,
        scoring=["f1", "precision", "recall"]
    )
    print(pd.DataFrame(scores).mean())

    model.fit(train_x, train_y)
    return model


def evaluate_model(model, val_x, val_y):
    predictions = model.predict(val_x)
    probabilities = model.predict_proba(val_x)
    y = np.asarray(val_y)

    metrics = {
        "precision": precision_score(y, predictions, zero_division=0),  # Of transactions flagged as scams, proportion actually scams
        "recall": recall_score(y, predictions, zero_division=0),  # Of all actual scams, proportion correctly detected
        "f1": f1_score(y, predictions, zero_division=0),  # Harmonic mean of precision and recall
        "log_loss": log_loss(y, probabilities),  # Penalises incorrect predicted probabilities; lower is better
        "roc_auc": roc_auc_score(y, probabilities[:, 1]),  # Measures how well scams are ranked above legitimate transactions
        "average_precision": average_precision_score(y, probabilities[:, 1])  # Summarises precision-recall performance across thresholds
    }

    # Bootstrap 95% confidence interval for F1
    # bootstrap_scores = []
    # for i in range(1000):
    #     idx = resample(np.arange(len(y)), replace=True, stratify=y, random_state=i)
    #     bootstrap_scores.append(f1_score(y[idx], predictions[idx], zero_division=0))

    # metrics["f1_ci_lower"], metrics["f1_ci_upper"] = np.percentile(bootstrap_scores, [2.5, 97.5])

    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)

    # Save feature importances or logistic regression coefficients
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = model.coef_[0]
    else:
        raise ValueError("Model does not expose feature importances or coefficients")

    pd.DataFrame({
        "feature": val_x.columns,
        "importance": importance
    }).sort_values("importance", ascending=False).to_csv(
        OUTPUT_DIR / "feature_importance.csv", index=False
    )

    # Save confusion matrix
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    pd.DataFrame(cm, index=["Actual legitimate", "Actual scam"],
                 columns=["Predicted legitimate", "Predicted scam"]).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    print(metrics)
    return metrics


def run_tree(train_x, train_y, val_x, val_y):
    model = train_model(train_x, train_y)
    metrics = evaluate_model(model, val_x, val_y)
    joblib.dump(model, OUTPUT_DIR / "decision_tree.joblib")
    print(metrics)
    return model