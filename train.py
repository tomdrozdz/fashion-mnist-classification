import os
import sys

# Disable tensorflow INFO messages, show only warnings and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
#################################################################

import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import plot_epochs
import matplotlib.pyplot as plt

# Parameters
classes = 10
split = 0.15
shape = (28, 28, 1)
epochs = 30
batch_size = 250
optimizer = keras.optimizers.Adam()
############


def cnn_model():
    cnn = keras.Sequential()
    
    cnn.add(keras.Input(shape=shape))
    
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Convolution2D(64, (3, 3), padding="same", activation="relu"))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Convolution2D(64, (3, 3), padding="same", activation="relu"))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn.add(keras.layers.Dropout(0.1))
    
    cnn.add(keras.layers.Convolution2D(128, (3, 3), padding="same", activation="relu"))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn.add(keras.layers.Dropout(0.2))
    
    cnn.add(keras.layers.Flatten())
    
    cnn.add(keras.layers.Dense(512, activation="relu"))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Dropout(0.3))
    cnn.add(keras.layers.Dense(256, activation="relu"))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Dropout(0.4))
    
    cnn.add(keras.layers.Dense(classes, activation="softmax", name="softmax"))

    cnn.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["acc"],
    )

    cnn.summary()
    return cnn


def train_model(
    model, x_train, y_train, x_val, y_val, epochs=epochs, batch_size=batch_size,
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
    from data import (
        get_data,
        load_data,
        get_labels,
        prepare_data,
        augument_data,
        split_data,
    )

    get_data("train")
    x_train, y_train = load_data("train")

    x_train, x_val, y_train, y_val = split_data(x_train, y_train)
    #x_train, y_train = augument_data(x_train, y_train)
    x_train = prepare_data(x_train)
    x_val = prepare_data(x_val)

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
