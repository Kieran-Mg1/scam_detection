from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay
)

OUTPUT_DIR = Path("results/test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_model(model, test_x, test_y, best_boundary):
    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= best_boundary).astype(int)

    metrics = {
        "boundary": best_boundary,
        "precision": precision_score(test_y, predictions, zero_division=0),
        "recall": recall_score(test_y, predictions, zero_division=0),
        "f1": f1_score(test_y, predictions, zero_division=0),
        "log_loss": log_loss(test_y, probabilities)
    }

    
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

    cm = confusion_matrix(test_y, predictions, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Scam"]).plot()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    print(metrics)
    return metrics