import pandas as pd
from pathlib import Path
from data.preprocess import data_preprocesser
from model.decision_tree import run_tree
from eval.opt_logit import opt_boundary
from eval.test import test_model

df = pd.read_csv("synsep_full_dataset.csv")

train_x, train_y, val_x, val_y, test_x, test_y = data_preprocesser(df)

model = run_tree(train_x, train_y, val_x, val_y)

boundary = opt_boundary(model, val_x, val_y)

test_model(model, test_x, test_y, boundary)

print('ran')