""" LSTM models for Q3 in Part 1 """
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential
from utils import load_train_test, reshape_data


# Define the LSTM models
def lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.5))
    model.add(LSTM(units=32))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[
                      tf.keras.metrics.AUC(name='auc'),
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall')
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
                      tf.keras.metrics.AUC(name='auc'),
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall')
                  ])
    return model


# Fit and evaluate models
def fit_evaluate(model, X_train, y_train, X_test, y_test):
    _ = model.fit(X_train, y_train,
                  epochs=50, batch_size=64,
                  validation_split=0.1)

    predictions = model.predict(X_test)

    roc_score = roc_auc_score(y_test, predictions)
    print(f"ROC-AUC: {roc_score:.3f}")

    precision, recall, _ = precision_recall_curve(y_test, predictions)
    auprc_score = auc(recall, precision)
    print(f"AUPRC: {auprc_score:.3f} \n")


if __name__ == "__main__":
    # Load the data
    dpath = Path("./data/")
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
