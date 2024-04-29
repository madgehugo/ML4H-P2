from pathlib import Path

from tensorflow.keras.utils import to_categorical

from ..part1.cnn import build_vanilla_cnn
from ..utils.utils import fit_evaluate, load_train_test, reshape_data

# Main
if __name__ == "__main__":
    # Load the data
    dpath = Path("./data/mitbih/")
    X_train, y_train, X_test, y_test = load_train_test(dpath)

    # Reshape the data for CNNs
    X_train_reshaped = reshape_data(X_train)
    X_test_reshaped = reshape_data(X_test)
    input_shape = (X_train_reshaped.shape[1], 1)

    # One-hot encode the target
    y_train_encoded = to_categorical(y_train, num_classes=5)
    y_test_encoded = to_categorical(y_test, num_classes=5)

    # Vanilla CNN
    print("--- Vanilla CNN ---")
    vanilla_cnn = build_vanilla_cnn(input_shape,
                                    loss='categorical_crossentropy',
                                    out_activation='softmax',
                                    num_classes=5)
    # TODO: fix choice of metrics (AUPRC doesn't work with multi-class)
    fit_evaluate(vanilla_cnn, X_train_reshaped, y_train_encoded,
                 X_test_reshaped, y_test_encoded)
