from pathlib import Path

from tensorflow.keras.utils import to_categorical

from src.part1.cnn import build_resnet_cnn
from src.utils.utils import fit_evaluate, load_train_test, reshape_data

# Main
if __name__ == "__main__":
    print("--- Transfer Learning ---")
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

    # Vanilla CNN
    # print("--- Vanilla CNN ---")
    # vanilla_cnn = build_vanilla_cnn(input_shape,
    #                                 loss='categorical_crossentropy',
    #                                 out_activation='softmax',
    #                                 num_classes=n_classes)
    # fit_evaluate(vanilla_cnn, X_train_reshaped, y_train_encoded,
    #              X_test_reshaped, y_test_encoded, num_classes=n_classes)

    # ResNet CNN
    print("--- ResNet CNN ---")
    resnet_cnn = build_resnet_cnn(input_shape,
                                  loss='categorical_crossentropy',
                                  out_activation='softmax',
                                  num_classes=n_classes)
    fit_evaluate(resnet_cnn, X_train_reshaped, y_train_encoded,
                 X_test_reshaped, y_test_encoded, num_classes=n_classes)
