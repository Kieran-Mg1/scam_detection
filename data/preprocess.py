import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler



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
    onehotcols = ['persona', 'country_type', 'remittance_category', 'day_of_week' , 'is_weekend', 'is_new_beneficiary', 'beneficiary_country', 'account_id']
    data = pd.get_dummies(data, columns=onehotcols, dtype=int)

    return data



def data_split(df):
    splits_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

    train_idx, test_idx = next(splits_train_test.split(df, df["is_fraud"]))
    trainval_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

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

    drop_cols = ['sender_iban','beneficiary_iban']
    float_cols = train_df.select_dtypes("float").columns

    scaler = StandardScaler()
    train_df[float_cols] = scaler.fit_transform(train_df[float_cols])
    val_df[float_cols] = scaler.transform(val_df[float_cols])
    test_df[float_cols] = scaler.transform(test_df[float_cols])

    # point in time correctness vs leakage is getting hairy, leaving for now
    # df["sender_previous_transfers"] = df.sort_values("timestamp").groupby("sender_iban").cumcount()
    # df["beneficiary_previous_transfers"] = df.sort_values("timestamp").groupby("beneficiary_iban").cumcount()
    
    train_df.drop(columns=drop_cols, inplace=True)
    val_df.drop(columns=drop_cols, inplace=True)
    test_df.drop(columns=drop_cols, inplace=True)

    return train_df, val_df, test_df