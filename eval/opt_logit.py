from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
OUTPUT_DIR = Path("results/opt_boundary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def opt_boundary(model, val_x, val_y, search_range=(0.01, 1.0, 1000), mode='C'):
    scores = []
    boundaries = np.linspace(*search_range)
    probabilities = model.predict_proba(val_x)[:, 1]
    y = np.asarray(val_y)
    amounts = pd.read_csv("val_amount.csv")["amount"].to_numpy()

    if mode == 'f1':
        for boundary in boundaries:
            class_pred = (probabilities >= boundary).astype(int)
            scores.append(f1_score(y, class_pred, zero_division=0))

        best_idx = np.argmax(scores)

    elif mode == 'C':
        for boundary in boundaries:
            class_pred = (probabilities >= boundary).astype(int)
            fn = (y == 1) & (class_pred == 0)
            fp = (y == 0) & (class_pred == 1)
            scores.append(amounts[fn].sum() + amounts[fp].sum())

        best_idx = np.argmin(scores)

    else:
        raise ValueError("mode must be 'f1' or 'C'")

    best_boundary = boundaries[best_idx]
    best_pred = (probabilities >= best_boundary).astype(int)

    fn = (y == 1) & (best_pred == 0)
    fp = (y == 0) & (best_pred == 1)
    cost = amounts[fn].sum() + amounts[fp].sum()
    f1 = f1_score(y, best_pred, zero_division=0)

    print(f"Best {mode} boundary: {best_boundary:.4f}, F1: {f1:.4f}, Cost: {cost:.2f}")

    pd.DataFrame([{"boundary": best_boundary, "f1": f1, "cost": cost}]).to_csv(
        OUTPUT_DIR / f"best_boundary_{mode}.csv", index=False
    )

    return best_boundary