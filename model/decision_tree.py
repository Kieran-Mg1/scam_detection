from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils import resample
import numpy as np

OUTPUT_DIR = Path("results/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_model(train_x, train_y):
    # unrestrained max depth makes the problem too easy. In most cases you'll have to limit the depth, so I do that here
    model = DecisionTreeClassifier(random_state=1, max_depth=10) 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    scores = cross_validate(model, train_x, train_y, cv=cv, scoring=["f1", "precision", "recall"])
    print(pd.DataFrame(scores).mean()) #should be close to val score

    model.fit(train_x, train_y)
    return model


def evaluate_model(model, val_x, val_y):
    predictions = model.predict(val_x)
    probabilities = model.predict_proba(val_x)
    
    metrics = {
        "precision": precision_score(val_y, predictions, zero_division=0),
        "recall": recall_score(val_y, predictions, zero_division=0),
        "f1": f1_score(val_y, predictions, zero_division=0),
        "log_loss": log_loss(val_y, probabilities)
    }
    bootstrap_scores = []

    for i in range(10):
        np_val_y = np.asarray(val_y)
        idx = resample(np.arange(len(np_val_y)), replace=True, stratify=np_val_y, random_state=i)
        bootstrap_scores.append(f1_score(np_val_y[idx], predictions[idx], zero_division=0))

    metrics["f1_ci_lower"], metrics["f1_ci_upper"] = np.percentile(bootstrap_scores, [2.5, 97.5]) # 95% confidence interval
    print(f"f1_ci_lower = {metrics['f1_ci_lower']:.3f}, f1_ci_upper = {metrics['f1_ci_upper']:.3f}")
    
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)

    cm = confusion_matrix(val_y, predictions, labels=[0, 1])
    pd.DataFrame(cm, index=["Actual legitimate", "Actual scam"], columns=["Predicted legitimate", "Predicted scam"]).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Scam"]).plot()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    return metrics


def run_tree(train_x, train_y, val_x, val_y):
    model = train_model(train_x, train_y)
    metrics = evaluate_model(model, val_x, val_y)
    joblib.dump(model, OUTPUT_DIR / "decision_tree.joblib")
    print(metrics)
    return model