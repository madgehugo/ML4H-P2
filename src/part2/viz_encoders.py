from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (auc, precision_recall_curve, roc_auc_score,
                             silhouette_score)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Conv1D, Dense, Dropout, Flatten, Input,
                                     MaxPooling1D, Reshape)
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


def load_train_test(dpath="./data/mitbih/"):
    df_train = pd.read_csv(dpath / 'train.csv', header=None)
    df_test = pd.read_csv(dpath / 'test.csv', header=None)

    # Train split
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    # Test split
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    return X_train, y_train, X_test, y_test


def reshape_data(X):
    return X.values.reshape((X.shape[0], X.shape[1], 1))


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


def build_resnet_cnn(input_shape, filters=32, kernel_size=5, strides=2,
                     loss='categorical_crossentropy', out_activation='softmax',
                     num_classes=5):
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
    outputs = Dense(num_classes, activation=out_activation)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy', AUC(name='auc'),
                           AUC(name='auprc', curve='PR'),
                           Precision(name='precision'),
                           Recall(name='recall')])
    return model


def fit_model(model, X_train, y_train, epochs=10,
              batch_size=64, val_split=0.1):
    model.fit(X_train, y_train,
              epochs=epochs, batch_size=batch_size,
              validation_split=val_split)


def extract_encoder(full_model, input_shape):
    encoder = Model(inputs=full_model.input,
                    outputs=full_model.get_layer('flatten').output)
    return encoder


def visualize_representations(encoder, dataset, labels, title, filename):
    # Obtain representations
    batch_size = 64
    num_batches = int(np.ceil(len(dataset) / batch_size))
    representations = []
    subset_labels = []

    for i in tqdm(range(num_batches), desc="Predicting"):
        batch_data = dataset[i * batch_size:(i + 1) * batch_size]
        batch_repr = encoder.predict(batch_data)
        representations.append(batch_repr)
        subset_labels.extend(labels[i * batch_size:(i + 1) * batch_size])

    representations = np.vstack(representations)
    subset_labels = np.array(subset_labels)

    # Dimensionality Reduction
    print("Performing PCA")
    pca = PCA(n_components=50)
    reduced_repr = pca.fit_transform(representations)

    print("Performing t-SNE")
    tsne = TSNE(n_components=2, random_state=42)
    reduced_repr = tsne.fit_transform(reduced_repr)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(reduced_repr, subset_labels)
    print(f'Silhouette Score: {silhouette_avg}')

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_repr[:, 0], y=reduced_repr[:, 1],
                    hue=subset_labels, palette='viridis', legend='full')
    plt.title(f'{title}\nSilhouette Score: {silhouette_avg:.2f}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Labels', loc='upper right')
    plt.savefig(filename)


def build_resnet_encoder(input_shape, filters=32, kernel_size=5,
                         strides=2, out_activation='sigmoid', num_classes=1):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters, kernel_size, strides=strides,
               padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = residual_block(x, filters, name='res_block1')
    x = MaxPooling1D(3, strides=strides, padding='same')(x)

    x = residual_block(x, 64, name='res_block2')
    x = MaxPooling1D(3, strides=strides, padding='same')(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation=out_activation)(x)

    encoder = Model(inputs, x, name='encoder')
    return encoder


def build_decoder(latent_dim, output_shape):

    encoded_input = Input(shape=(latent_dim,))
    x = Dense(64, activation='relu')(encoded_input)
    x = Dropout(0.5)(x)  # Add dropout layer with a dropout rate of 0.5
    x = Dense(output_shape[0] * output_shape[1], activation='relu')(x)
    x = Reshape(output_shape)(x)

    decoder = Model(encoded_input, x, name='decoder')
    return decoder


# Fit and evaluate models
def fit_evaluate(model, X_train, y_train, X_test, y_test,
                 epochs=50, batch_size=64, val_split=0.1,
                 num_classes=1):

    _ = model.fit(X_train, y_train,
                  epochs=epochs, batch_size=batch_size,
                  validation_split=val_split)

    predictions = np.array(model.predict(X_test))

    roc_score = roc_auc_score(y_test, predictions, multi_class='ovo')
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
    # ----------- MITBIH ----------- #
    print("--- ResNet Encoder with Visualization ---")
    # Load the data
    mitbih_dpath = Path("./data/mitbih/")
    X_train_mitbih, y_train_mitbih, X_test_mitbih, y_test_mitbih = load_train_test(mitbih_dpath)

    # Reshape the data for CNNs
    X_train_mitbih_reshaped = reshape_data(X_train_mitbih)
    X_test_mitbih_reshaped = reshape_data(X_test_mitbih)
    input_shape = (X_train_mitbih.shape[1], 1)

    # Select every 10th datapoint
    X_train_mitbih_subset = reshape_data(X_train_mitbih.iloc[::10])
    y_train_mitbih_subset = y_train_mitbih.iloc[::10]

    # One-hot encode the target
    n_classes = 5
    y_train_mitbih_encoded = to_categorical(y_train_mitbih,
                                            num_classes=n_classes)
    y_test_mitbih_encoded = to_categorical(y_test_mitbih,
                                           num_classes=n_classes)

    # ----------- PTBDB ----------- #
    # Load the data
    ptbdb_dpath = Path("./data/ptbdb/")
    X_train_ptbdb, y_train_ptbdb, X_test_ptbdb, y_test_ptbdb = load_train_test(ptbdb_dpath)

    # Select every 10th datapoint
    X_train_ptbdb_subset = reshape_data(X_train_ptbdb.iloc[::3])
    y_train_ptbdb_subset = y_train_ptbdb.iloc[::3]

    #######################################################

    # Build and train the full ResNet model
    resnet_model = build_resnet_cnn(input_shape, num_classes=n_classes)
    # Changed from fit_model - to check any differences to cnn_transfer
    fit_evaluate(resnet_model, X_train_mitbih_reshaped, y_train_mitbih_encoded,
                 X_test_mitbih_reshaped, y_test_mitbih_encoded, epochs=10)

    # Extract the encoder from the trained ResNet model
    resnet_encoder = extract_encoder(resnet_model, input_shape)

    # Visualize learned representations subset of the MIT-BIH dataset
    visualize_representations(resnet_encoder, X_train_mitbih_subset,
                              y_train_mitbih_subset,
                              'Learned Representations - MIT-BIH (Subset)',
                              "./results/RESNET-MITBIH.png")
    visualize_representations(resnet_encoder, X_train_ptbdb_subset,
                              y_train_ptbdb_subset,
                              'Learned Representations - PTBDB (Subset)',
                              "./results/RESNET-PTBDB.png")

    # AUTOENCODER STUFF #
    input_shape = (X_train_mitbih_reshaped.shape[1], 1)
    encoding_dim = 64

    encoder = build_resnet_encoder(input_shape, filters=32, kernel_size=5,
                                   strides=2, out_activation='sigmoid',
                                   num_classes=64)
    decoder = build_decoder(64, input_shape)

    autoencoder_input = Input(shape=input_shape)
    encoded_output = encoder(autoencoder_input)
    decoded_output = decoder(encoded_output)

    autoencoder = Model(autoencoder_input, decoded_output, name='autoencoder')
    autoencoder.compile(
        optimizer='adam', loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'),
                 AUC(name='auprc', curve='PR'),
                 Precision(name='precision'),
                 Recall(name='recall')])

    autoencoder.fit(X_train_mitbih_reshaped, X_train_mitbih_reshaped,
                    epochs=10,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.1)

    visualize_representations(encoder, X_train_mitbih_subset,
                              y_train_mitbih_subset,
                              'Learned Representations - MIT-BIH (Subset)',
                              "./results/Autoencoder-MITBIH.png")
    visualize_representations(encoder, X_train_ptbdb_subset,
                              y_train_ptbdb_subset,
                              'Learned Representations - PTBDB (Subset)',
                              "./results/Autoencoder-PTBDB.png")
    print(X_train_mitbih_subset.shape, y_train_mitbih_subset.shape)
    # Plot both datasets vis t-SNE | color = dataset
    X_train_combined = np.vstack((X_train_mitbih_subset, X_train_ptbdb_subset))
    mitbih_labels = np.zeros((X_train_mitbih_subset.shape[0], 1))  # 0 for mitbih
    ptbdb_labels = np.ones((X_train_ptbdb_subset.shape[0], 1))     # 1 for ptbdb
    print(mitbih_labels.shape, ptbdb_labels.shape)
    # Stack the binary labels vertically
    y_indicator = np.vstack((mitbih_labels, ptbdb_labels))
    y_indicator = y_indicator.flatten()
    print(X_train_combined.shape, y_indicator.shape)
    visualize_representations(resnet_encoder, X_train_combined,
                              y_indicator,
                              'Comparison of dataset embeddings (RESNET)',
                              "./results/RESNET-both.png")
    visualize_representations(encoder, X_train_combined,
                              y_indicator,
                              'Comparison of dataset embeddings (AUTOENCODER)',
                              './results/Autoencoder-both.png')
