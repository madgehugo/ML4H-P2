from pathlib import Path
from warnings import filters

from tensorflow.keras.utils import to_categorical

# from src.part1.cnn import build_resnet_cnn
# from src.utils.utils import fit_evaluate, load_train_test, reshape_data

from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Conv1D, Dense, Dropout, Flatten, Input,
                                     MaxPooling1D)
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, Dense, Dropout, UpSampling1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Reshape, Conv1DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
import keras
from keras import layers


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

# Define the residual block
def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    if conv_shortcut:
        shortcut = Conv1D(filters, 1, strides=stride, name=name+'_0_conv')(x)
        shortcut = BatchNormalization(name=name+'_0_bn')(shortcut)
    else:
        shortcut = x

    x = Conv1D(filters, kernel_size, padding='same', strides=stride, kernel_regularizer=l2(0.001), name=name+'_1_conv')(x)
    x = BatchNormalization(name=name+'_1_bn')(x)
    x = Activation('relu', name=name+'_1_relu')(x)

    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001), name=name+'_2_conv')(x)
    x = BatchNormalization(name=name+'_2_bn')(x)

    x = Add()([shortcut, x])
    x = Activation('relu', name=name+'_out')(x)
    return x

# Define the encoder
def build_resnet_encoder(input_shape, filters=32, kernel_size=5, strides=2, out_activation='sigmoid',
                     num_classes=1):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters, kernel_size, strides=strides, padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    print("ACTIVATION")
    print(x.shape)

    x = residual_block(x, filters, name='res_block1')
    x = MaxPooling1D(3, strides=strides, padding='same')(x)
    print("RES BLOCK")
    print(x.shape)

    x = residual_block(x, 64, name='res_block2')
    x = MaxPooling1D(3, strides=strides, padding='same')(x)
    print("RES BLOCK")
    print(x.shape)

    x = Flatten()(x)
    print("FLFATTEN")
    print(x.shape)
    
    x = Dense(64, activation='relu')(x)
    print("DENSE")
    print(x.shape)
    x = Dropout(0.5)(x)
    print("DROPOUT")
    print(x.shape)
    x = Dense(num_classes, activation=out_activation)(x)
    print(x.shape)

    encoder = Model(inputs, x, name='encoder')


    return encoder

# Define the decoder
def build_decoder_1(latent_dim, original_shape):

    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(187, activation='sigmoid')(decoded)
    return decoded



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



# Main
if __name__ == "__main__":
    print("--- Representation Learning Q2.2 ---")
    # Load the data
    dpath = Path("./data/mitbih/")
    X_train, y_train, X_test, y_test = load_train_test(dpath)

    # Reshape the data for CNNs
    X_train_reshaped = reshape_data(X_train)
    X_test_reshaped = reshape_data(X_test)
    input_shape = (X_train_reshaped.shape[1], 1)
    
    # One-hot encode the target
    n_classes = 5
    y_train_encoded = to_categorical(y_train, num_classes=n_classes)
    y_test_encoded = to_categorical(y_test, num_classes=n_classes)

    latent_dim=64
    encoded = build_resnet_encoder(input_shape, filters=32, kernel_size=5, strides=2, out_activation='sigmoid',
                     num_classes=1)

    decoded = build_decoder_1(64, 187)

    autoencoder = keras.Model(input_shape, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(X_train_reshaped, X_train_reshaped,
                epochs=100,
                batch_size=256,
                shuffle=True)



  