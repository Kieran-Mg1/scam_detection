import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def data_preprocesser(data):
    '''
    step1: one hot encode catagorical vars, ensure correct types
    step2: split data into train, val, test.
    ste3: remove and seperate labels.
    step4: process columms, it is safe to normalise now because we have split the data already.

    '''
    # step 1
    data = format_col_types(data)
    # step 2
    train_df, val_df, test_df = data_split(data)
    # step 3
    train_df_preproc, train_y, val_df_preproc, val_y, test_df_preproc, test_y = xy_select(train_df, val_df, test_df)
    #step 4
    train_x, val_x, test_x = standardise_cols(train_df_preproc, val_df_preproc, test_df_preproc)

    for df in [train_x, val_x, test_x]:
        non_numeric = df.select_dtypes(exclude="number").columns.tolist()
        assert not non_numeric, f"Non-numeric columns found: {non_numeric}"

    return train_x, train_y, val_x, val_y, test_x, test_y


def format_col_types(data):
    onehotcols = ['country_type', 'day_of_week' , 'is_weekend', 'is_new_beneficiary', 'beneficiary_country']
    drop_cols = ['sender_iban','beneficiary_iban', 'account_id', 'remittance_text', 'persona', 'remittance_category'] #persona, remittance_category imo could be a class label
    # convert time to linux time
    data["timestamp"] = pd.to_datetime(data["timestamp"]).astype("int64") / 1e9
    # convert time to onehot
    data = pd.get_dummies(data, columns=onehotcols, dtype=int)
    # does not have to be inplace, but I prefer over constant reassignment
    data.drop(columns=drop_cols, inplace=True)

    return data


def data_split(df):
    #split trainval 80 / test 20
    splits_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

    # get indices
    train_idx, test_idx = next(splits_train_test.split(df, df["is_fraud"]))
    trainval_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    #split train 60 / val 20
    splits_train_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=1)

    train_idx, val_idx = next(splits_train_val.split(trainval_df, trainval_df["is_fraud"]))
    val_df = trainval_df.iloc[val_idx]
    train_df = trainval_df.iloc[train_idx]

    
    return train_df, val_df, test_df

def xy_select(train_df, val_df, test_df):
    target = 'is_fraud'
    drop_cols = ['is_fraud', 'fraud_type', 'transaction_id']

    test_y = test_df[target]
    test_x = test_df.drop(columns=drop_cols)

    val_y = val_df[target]
    val_x = val_df.drop(columns=drop_cols)

    train_y = train_df[target]
    train_x = train_df.drop(columns=drop_cols)
    
    return train_x, train_y, val_x, val_y, test_x, test_y


def standardise_cols(train_df, val_df, test_df):
    float_cols = train_df.select_dtypes("float").columns

    transforms = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()) # won't matter for a tree, but should improve gradient stability in logreg, nn, ..
    ])

    train_df[float_cols] = transforms.fit_transform(train_df[float_cols])
    val_df[float_cols] = transforms.transform(val_df[float_cols])
    test_df[float_cols] = transforms.transform(test_df[float_cols])

    return train_df, val_df, test_df