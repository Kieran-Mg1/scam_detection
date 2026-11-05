from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
OUTPUT_DIR = Path("results/opt_boundary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def opt_boundary(model, val_x, val_y, search_range=(0.01, 1.0, 1000)):

    f1_scores = []
    C_scores = []

    boundaries = np.linspace(*search_range)
    probabilites = model.predict_proba(val_x)[:, 1]
    print('prob range:', pd.Series(model.predict_proba(val_x)[:, 1]).value_counts().iloc[:5])
    
    for boundary in boundaries:
        
        class_pred = (probabilites >= boundary).astype(int)
        f1 = f1_score(val_y, class_pred, zero_division=0)
        C_score = ... #todo

        f1_scores.append(f1) 

    best_boundary_f1 = boundaries[np.argmax(f1_scores)]
    print('best boundary:', best_boundary_f1, f1_scores[np.argmax(f1_scores)])
    pd.DataFrame([best_boundary_f1]).to_csv(OUTPUT_DIR / "best_boundary_f1.csv", index=False)
    return best_boundary_f1