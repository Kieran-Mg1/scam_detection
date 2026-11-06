from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    f1_score,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score
)

OUTPUT_DIR = Path("results/test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_model(model, test_x, test_y, best_boundary, min_recall=0.85, max_fpr=0.001):
    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= best_boundary).astype(int)

    cm = confusion_matrix(test_y, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    metrics = {
        "boundary": best_boundary,  # Decision threshold selected using validation data
        "precision": precision_score(test_y, predictions, zero_division=0),  # Of transactions flagged as scams, proportion actually scams
        "recall": recall,  # Of all actual scams, proportion correctly detected
        "fpr": fpr,  # Of legitimate transactions, proportion incorrectly flagged as scams
        "f1": f1_score(test_y, predictions, zero_division=0),  # Harmonic mean of precision and recall
        "log_loss": log_loss(test_y, probabilities),  # Penalises incorrect predicted probabilities; lower is better
        "roc_auc": roc_auc_score(test_y, probabilities),  # Measures how well scams are ranked above legitimate transactions
        "average_precision": average_precision_score(test_y, probabilities),  # Summarises precision-recall performance across thresholds
        "constraints_met": recall >= min_recall and fpr <= max_fpr  # Whether both business requirements are satisfied
    }

    output_dir = Path("results/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Scam"]).plot()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    print(metrics)
    return metrics