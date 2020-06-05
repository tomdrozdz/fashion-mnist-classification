import os
import sys

# Disable tensorflow INFO messages, only show warnings and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
#################################################################

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from utils import plot_epochs
import matplotlib.pyplot as plt

# Parameters
classes = 10
default_split = 0.15
default_shape = (28, 28, 1)
default_epochs = 80
default_batch_size = 250
default_optimizer = keras.optimizers.Adam()
############


def split_data(x, y, split=default_split):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split)
    return x_train, x_val, y_train, y_val


def cnn_model(
    layers=[], optimizer=default_optimizer, name=None, data_shape=default_shape
):

    layers = [keras.Input(shape=data_shape)] + layers

    if len(layers) == 1:
        layers.append(keras.layers.BatchNormalization())
        layers.append(keras.layers.Convolution2D(64, (4, 4), padding="same", activation="relu"))
        # layers.append(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        layers.append(keras.layers.Dropout(0.1))
        layers.append(keras.layers.Convolution2D(64, (4, 4), activation="relu"))
        # layers.append(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        layers.append(keras.layers.Dropout(0.3))
        layers.append(keras.layers.Flatten())
        layers.append(keras.layers.Dense(256, activation="relu"))
        layers.append(keras.layers.Dropout(0.5))
        layers.append(keras.layers.Dense(64, activation="relu"))
        layers.append(keras.layers.BatchNormalization())

    layers.append(keras.layers.Dense(classes, activation="softmax", name="softmax"))

    cnn = keras.Sequential(layers=layers, name=name)

    cnn.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["acc"],
    )

    cnn.summary()
    return cnn


def train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=default_epochs,
    batch_size=default_batch_size,
):
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
    )


def plot_history(model):
    history = model.history.history

    plt.figure(num="Training history", figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plot_epochs(
        "Model accuracy",
        [history["acc"], history["val_acc"]],
        "Accuracy",
        ["acc", "val_acc"],
        "upper left",
    )

    plt.subplot(1, 2, 2)
    plot_epochs(
        "Model loss",
        [history["loss"], history["val_loss"]],
        "Loss",
        ["loss", "val_loss"],
        "upper right",
    )

    plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress(0)


if __name__ == "__main__":
    print("Running training for the default model")
    from data import get_data, load_data, get_labels, prepare_data, augument_data

    get_data("train")
    x_train, y_train = load_data("train")

    # y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)

    shape = (-1, 28, 28, 1)
    
    x_train, y_train = augument_data(x_train, y_train)
    x_train = prepare_data(x_train, shape)
    x_train, x_val, y_train, y_val = split_data(x_train, y_train)

    model = cnn_model()
    train_model(model, x_train, y_train, x_val, y_val)

    plot_history(model)

    from models import save_model_h5

    if len(sys.argv) == 2:
        name = ""
    else:
        name = input(
            f'Enter model name (leaving blank saves model as "latest"): '
        ).strip()

    if name != "":
        save_model_h5(model, name)
    else:
        save_model_h5(model)
