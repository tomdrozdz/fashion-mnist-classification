import os

# Disable tensorflow INFO messages, only show warnings and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
#################################################################
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

default_val_data_split = 0.15


def split_data(x, y, split=default_val_data_split):
    x_val = None
    y_val = None
    return x_train, y_train, x_val, y_val


def cnn_model(data_shape=(28, 28)):
    cnn = keras.Sequential()

    cnn.add(keras.layers.Flatten(input_shape=data_shape))
    cnn.add(keras.layers.Dense(256, activation="relu"))
    cnn.add(keras.layers.Dense(64, activation="relu"))
    cnn.add(keras.layers.Dense(10, activation="softmax"))

    cnn.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    cnn.summary()

    return cnn


def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
    )


def plot_history(model):
    history = model.history.history

if __name__ == "__main__":
    print("Running training for the default model")

    from data import get_all_data, load_all_data, get_labels

    get_all_data()
    (x_train, y_train), (x_test, y_test) = load_all_data()
    labels = get_labels()

    # Normalization 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    # show_item(x_train, y_train, np.random.randint(len(x_train)))

    epochs = 10
    batch_size = 250

    model = cnn_model()
    train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size)

    from models import save_model_h5

    name = input(f'Enter (leaving blank saves model as "latest"): ').strip()

    if name != "":
        save_model_h5(model, name)
    else:
        save_model_h5(model)
