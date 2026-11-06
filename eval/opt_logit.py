from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

OUTPUT_DIR = Path("results/opt_boundary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def opt_boundary(model, val_x, val_y, search_range=(0.001, 1.0, 1000),
                 mode='constraints', min_recall=0.85, max_fpr=0.001):

    scores = []
    boundaries = np.linspace(*search_range)
    probabilities = model.predict_proba(val_x)[:, 1]
    y = np.asarray(val_y)

    if mode == 'f1':
        for boundary in boundaries:
            # Classify transactions as scams (True) if probability >= boundary
            pred = probabilities >= boundary

            # Calculate F1 at this boundary and store it
            scores.append(f1_score(y, pred, zero_division=0))

        # Find the index of the boundary with the highest F1
        best_idx = np.argmax(scores)

    elif mode == 'C':
        amounts = pd.read_csv("val_amount.csv")["amount"].to_numpy()

        for boundary in boundaries:
            # Classify transactions as scams if probability >= boundary
            pred = probabilities >= boundary

            # Identify actual scams incorrectly predicted as legitimate
            fn = (y == 1) & ~pred

            # Identify legitimate transactions incorrectly predicted as scams
            fp = (y == 0) & pred

            # Sum transaction amounts for all errors and store the total cost
            scores.append(amounts[fn | fp].sum())

        # Find the index of the boundary with the lowest cost
        best_idx = np.argmin(scores)

    elif mode == 'constraints':
        recall_list = []
        fpr_list = []
        for boundary in boundaries:
            # Classify transactions as scams if probability >= boundary
            pred = probabilities >= boundary

            # Proportion of actual scams correctly detected
            recall = recall_score(y, pred, zero_division=0)

            # Proportion of legitimate transactions incorrectly flagged as scams
            fpr = ((y == 0) & pred).sum() / (y == 0).sum()

            # Keep recall if both business constraints are satisfied; otherwise store -1
            scores.append(recall if recall >= min_recall and fpr <= max_fpr else -1)
            recall_list.append(recall)
            fpr_list.append(fpr)

        # Check whether any boundary satisfied both constraints
        if max(scores) == -1:
            raise ValueError("No boundary satisfies both constraints")

        # Select the valid boundary that achieves the highest recall
        best_idx = np.argmax(scores)

    else:
        raise ValueError("mode must be 'f1', 'C' or 'constraints'")

    best_boundary = boundaries[best_idx]
    print(f"Best {mode} boundary: {best_boundary:.4f}, score: {scores[best_idx]:.4f}")

    pd.DataFrame([{"boundary": best_boundary, "score": scores[best_idx]}]).to_csv(
        OUTPUT_DIR / f"best_boundary_{mode}.csv", index=False
    )

    return best_boundary