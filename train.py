import os

# Disable tensorflow INFO messages, only show warnings and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
#################################################################
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from utils import normalize
import matplotlib.pyplot as plt


def split_data(x, y, split=0.15):
    return train_test_split(x, y, test_size=split)


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
        # validation_data=(x_val, y_val),
    )


def plot_history(model):
    history = model.history.history


if __name__ == "__main__":
    print("Running training for the default model")
    from data import get_data, load_data, get_labels

    get_data("train")
    x_train, y_train = load_data("train")

    # Parameters
    data_split = 0.15
    epochs = 10
    batch_size = 250

    x_train = normalize(x_train)
    x_train, y_train, x_val, y_val = split_data(x_train, y_train, data_split)

    model = cnn_model()
    train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size=batch_size)

    from models import save_model_h5

    name = input(f'Enter model name (leaving blank saves model as "latest"): ').strip()

    if name != "":
        save_model_h5(model, name)
    else:
        save_model_h5(model)
