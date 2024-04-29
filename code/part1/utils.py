import pandas as pd


# TODO: Add headers to the data
def load_train_test():
    df_train = pd.read_csv('../../data/ptbdb_train.csv', header=None)
    df_test = pd.read_csv('../../data/ptbdb_test.csv', header=None)

    # Train split
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    # Test split
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    return X_train, y_train, X_test, y_test
