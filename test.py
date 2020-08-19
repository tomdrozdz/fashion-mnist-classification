import os
import sys

# Disable tensorflow INFO messages, only show warning and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
################################################################
import tensorflow as tf
from tensorflow import keras
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from utils import plot_image_prediction, plot_predictions
import matplotlib.pyplot as plt


def predict_classes(model, x_test, y_test):
    loss, _ = model.evaluate(x_test, y_test, batch_size=250)
    predictions = model.predict(x_test)
    y_pred = predictions.argmax(axis=1)
    accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
    return y_pred, predictions, accuracy, loss


def show_errors(x_test, y_test, y_pred, predictions, labels):
    rows = 4
    cols = 3
    imgs = rows * cols

    idxs = np.where(y_test != y_pred)[0]
    imgs = imgs if imgs <= len(idxs) else len(idxs)
    idxs = np.random.choice(idxs, imgs)

    x_test = x_test[idxs]
    y_test = y_test[idxs]
    y_pred = y_pred[idxs]
    predictions = predictions[idxs]

    x_test = x_test.reshape(-1, 28, 28)

    plt.figure(
        num="Examples of incorrect classifications", figsize=(2 * 2 * cols, 2 * rows)
    )

    for i in range(imgs):
        y_t, y_p, pred = y_test[i], y_pred[i], predictions[i]

        plt.subplot(rows, 2 * cols, 2 * i + 1)
        plot_image_prediction(
            x_test[i], labels[y_t], labels[y_p], pred[y_p], pred[y_t],
        )
        plt.subplot(rows, 2 * cols, 2 * i + 2)
        plot_predictions(y_t, y_p, pred)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, labels):
    plt.figure(num="Confusion matrix", figsize=(8, 8))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    threshold = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from models import load_model_h5

    if len(sys.argv) >= 2:
        file_name = sys.argv[1]
        print("Running testing for", file_name)
        model = load_model_h5(file_name)
    else:
        print("Running the default testing")
        model = load_model_h5()

    from data import get_data, load_data, get_labels, prepare_data

    get_data("t10k")
    x_test, y_test = load_data("t10k")
    labels = get_labels()

    x_test = prepare_data(x_test)

    y_pred, predictions, accuracy, loss = predict_classes(model, x_test, y_test)

    print("\nModel accuracy:", accuracy)
    print(f"Value of the loss function: {loss:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels)

    show_errors(x_test, y_test, y_pred, predictions, labels)
