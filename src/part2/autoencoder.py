from pathlib import Path
from secrets import randbelow

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, Dense, Dropout, UpSampling1D, Reshape, Add
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


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


# RES BLOCK TASK 1
def residual_block(x, filters, kernel_size=3, strides=1, name='name'):
    # Shortcut connection
    shortcut = x

    # First convolution layer
    x = Conv1D(filters, kernel_size, strides=strides, padding='same', name=name+'_conv1')(x)
    x = BatchNormalization(name=name+'_bn1')(x)
    x = Activation('relu')(x)

    # Second convolution layer
    x = Conv1D(filters, kernel_size, strides=1, padding='same', name=name+'_conv2')(x)
    x = BatchNormalization(name=name+'_bn2')(x)

    # Add shortcut to the output
    x = x + shortcut
    x = Activation('relu')(x)

    return x


if __name__ == "__main__":
    print("--- Representation Learning Q2.2 ---")
    # Load the data
    dpath = Path("../../data/mitbih/")
    X_train, y_train, X_test, y_test = load_train_test(dpath)

    # Reshape the data for CNNs
    X_train_reshaped = reshape_data(X_train)
    X_test_reshaped = reshape_data(X_test)
    input_dim = X_train_reshaped.shape[1]
    # input_shape = (X_train_reshaped.shape[1], 1)
    encoding_dim = 64

    input_layer = Input(shape=(input_dim, 1))

    x = Flatten()(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(x)

    # Decoder
    x = Dense(input_dim, activation='sigmoid')(encoded)
    decoded = Reshape((input_dim, 1))(x)

    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Summary of the model
    autoencoder.summary()

    autoencoder.fit(X_train_reshaped, X_train_reshaped, epochs=5, batch_size=256, shuffle=True, validation_split=0.2)
