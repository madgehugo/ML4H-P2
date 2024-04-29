from pathlib import Path

from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Conv1D, Dense, Dropout, Flatten, Input,
                                     MaxPooling1D)
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from utils import fit_evaluate, load_train_test, reshape_data


# Vanilla CNN
def build_vanilla_cnn(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3,
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3,
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[
                      AUC(name='auc'),
                      Precision(name='precision'),
                      Recall(name='recall')
                  ])
    return model


# ResNet CNN
def residual_block(x, filters, kernel_size=3, stride=1,
                   conv_shortcut=True, name=None):
    # Shortcut
    if conv_shortcut is True:
        shortcut = Conv1D(filters, 1, strides=stride, name=name+'_0_conv')(x)
        shortcut = BatchNormalization(name=name+'_0_bn')(shortcut)
    else:
        shortcut = x

    # Residual
    x = Conv1D(filters, kernel_size, padding='same', strides=stride,
               kernel_regularizer=l2(0.001), name=name+'_1_conv')(x)
    x = BatchNormalization(name=name+'_1_bn')(x)
    x = Activation('relu', name=name+'_1_relu')(x)

    x = Conv1D(filters, kernel_size, padding='same',
               kernel_regularizer=l2(0.001), name=name+'_2_conv')(x)
    x = BatchNormalization(name=name+'_2_bn')(x)

    # Add shortcut
    x = Add()([shortcut, x])
    x = Activation('relu', name=name+'_out')(x)
    return x


def build_resnet_cnn(input_shape, filters=32, kernel_size=5, strides=2):
    inputs = Input(shape=input_shape)

    # Initial conv block
    x = Conv1D(filters, kernel_size, strides=strides,
               padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # Residual blocks
    x = residual_block(x, filters, name='res_block1')
    x = MaxPooling1D(3, strides=strides, padding='same')(x)

    x = residual_block(x, 64, name='res_block2')
    x = MaxPooling1D(3, strides=strides, padding='same')(x)

    # Final layers
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, x)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Main
if __name__ == "__main__":
    # Load the data
    dpath = Path("./data/")
    X_train, y_train, X_test, y_test = load_train_test(dpath)

    # Reshape the data for CNNs
    X_train_reshaped = reshape_data(X_train)
    X_test_reshaped = reshape_data(X_test)
    input_shape = (X_train_reshaped.shape[1], 1)

    # Vanilla CNN
    print("--- Vanilla CNN ---")
    vanilla_cnn = build_vanilla_cnn(input_shape)
    fit_evaluate(vanilla_cnn, X_train_reshaped, y_train,
                 X_test_reshaped, y_test)

    # ResNet CNN
    print("--- ResNet CNN ---")
    resnet_cnn = build_resnet_cnn(input_shape)
    fit_evaluate(resnet_cnn, X_train_reshaped, y_train,
                 X_test_reshaped, y_test)
