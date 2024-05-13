""" LSTM models for Q3 in Part 1 """
from pathlib import Path

from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Sequential

from src.utils.utils import fit_evaluate, load_train_test, reshape_data


# Define the LSTM models
def log_reg_model(X_train):
    model = Sequential()
    model.add(
        Dense(
            2, 
            activation='softmax',
            kernel_regularizer=L1L2(l1=0.0, l2=0.1),
            input_dim=len(feature_vector)
            )) # input dimension = number of features your data has
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall')
        ])
    return model


def lstm_model_bidirectional(X_train):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=64, return_sequences=True),
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(units=32)))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[
                      AUC(name='auc'),
                      Precision(name='precision'),
                      Recall(name='recall')
                  ])
    return model


if __name__ == "__main__":
    # Load the data
    dpath = Path("./data/ptbdb/")
    X_train, y_train, X_test, y_test = load_train_test(dpath)

    # Reshape the data for LSTM
    X_train_LSTM = reshape_data(X_train)
    X_test_LSTM = reshape_data(X_test)

    # LSTM
    print("--- LSTM ---")
    lstm = lstm_model(X_train_LSTM)
    fit_evaluate(lstm, X_train_LSTM, y_train, X_test_LSTM, y_test)

    # Bidirectional LSTM
    print("--- Bidirectional LSTM ---")
    lstm_bi = lstm_model_bidirectional(X_train_LSTM)
    fit_evaluate(lstm_bi, X_train_LSTM, y_train, X_test_LSTM, y_test)
