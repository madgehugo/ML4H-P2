from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Sequential

import tensorflow_decision_forests as tfdf

from tsfresh import extract_features



#from src.utils.utils import fit_evaluate, load_train_test, reshape_data
# TEMP as I had import issues with src utils

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import label_binarize


# TODO: Add headers to the data
def load_train_test(dpath="../../data/ptbdb/"):
    df_train = pd.read_csv(dpath / 'train.csv', header=None)
    df_test = pd.read_csv(dpath / 'test.csv', header=None)

    # Train split
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    # Test split
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    return X_train, y_train, X_test, y_test


# Reshape the data for LSTM
def reshape_data(X):
    return X.values.reshape((X.shape[0], X.shape[1], 1))


# Fit and evaluate models
def fit_evaluate(model, X_train, y_train, X_test, y_test,
                 epochs=50, batch_size=64, val_split=0.1,
                 num_classes=1):

    _ = model.fit(X_train, y_train,
                  epochs=epochs, batch_size=batch_size,
                  validation_split=val_split)

    predictions = model.predict(X_test)

    roc_score = roc_auc_score(y_test, predictions)
    print(f"ROC-AUC: {roc_score:.3f}")

    if num_classes == 1:
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        auprc_score = auc(recall, precision)
        print(f"AUPRC: {auprc_score:.3f} \n")

    else:
        # One vs. Rest (OvR) AUPRC
        y_test_binarized = label_binarize(
            y_test, classes=np.arange(num_classes)
            )

        # Calculate AUPRC for each class
        auprc_scores = []
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(
                y_test_binarized[:, i],
                predictions[:, i]
                )
            auprc_score = auc(recall, precision)
            auprc_scores.append(auprc_score)

        # Calculate the average AUPRC across all classes
        average_auprc = np.mean(auprc_scores)

        print("Average AUPRC: {:.3f}".format(average_auprc))


def log_reg_model(X_train):
    model = Sequential()
    model.add(
        Dense(
            1, 
            activation='sigmoid',  # Sigmoid activation for logistic regression
            input_shape=(X_train.shape[1],)  # Input shape is 1D
            )) 
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall')
        ])
    return model


import keras
from tensorflow.keras import layers
import tensorflow as tf
print(tf.__version__)
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x, weights = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout, return_attention_scores=True
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


if __name__ == "__main__":
    # Load the data
    dpath = Path("../../data/ptbdb/")
    X_train, y_train, X_test, y_test = load_train_test(dpath)

    # Reshape the data for LSTM
    X_train = reshape_data(X_train)
    X_test = reshape_data(X_test)

    print(X_test[:1].shape)
    print(y_test[:1].shape)

    inputs = keras.Input(shape=(X_train.shape[1],1))
    x = inputs
    x = transformer_encoder(x, 64, 4, 4, 0.1)
    x = layers.GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="selu")(x)
    x = Dense(32, activation="selu")(x)
    x = Dense(16, activation="selu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    tf_model= keras.Model(inputs, outputs)

    tf_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall')
        ])
    print(tf_model.summary())
    print(tf_model.layers[1])
    tf_model.fit(X_train, y_train, validation_split=0.2, epochs=10)
    scores = tf_model.evaluate(X_test,y_test)

    predictions = tf_model.predict(X_test)

    roc_score = roc_auc_score(y_test, predictions)
    print(f"ROC-AUC: {roc_score:.3f}")

    precision, recall, _ = precision_recall_curve(y_test, predictions)
    auprc_score = auc(recall, precision)
    print(f"AUPRC: {auprc_score:.3f} \n")

    import seaborn as sb
    import matplotlib.pyplot as plt
    attention_layer = tf_model.layers[1]
    _, attention_scores = attention_layer(y_test[:1], X_test[:1], return_attention_scores=True) # take one sample
    fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[5,5,0.2]))
    sb.heatmap(attention_scores[0, 0, :, :], annot=True, cbar=False, ax=axs[0])
    sb.heatmap(attention_scores[0, 1, :, :], annot=True, yticklabels=False, cbar=False, ax=axs[1])
    fig.colorbar(axs[1].collections[0], cax=axs[2])
    plt.show()
    