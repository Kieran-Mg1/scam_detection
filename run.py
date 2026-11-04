import pandas as pd
from pathlib import Path
from data.preprocess import data_preprocesser

df = pd.read_csv("synsep_full_dataset.csv")

train_x, train_y, val_x, val_y, test_x, test_y = data_preprocesser(df)

print('ran')