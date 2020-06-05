import matplotlib.pyplot as plt
import numpy as np

# Normalization 0-1
def normalize(x):
    return x / 255


# For plotting charts based on epochs
def plot_epochs(title, values, y_label, legend, loc):
    epochs = len(values[0])
    x = np.arange(start=1, stop=epochs + 1)

    for series in values:
        plt.plot(x, series)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.xlim([1, epochs])
    plt.legend(legend, loc=loc)


# Utility for displaying the images
def show_image(img, label):
    plt.imshow(img, cmap="gray")
    plt.title(label)
    plt.axis(False)


# For displaying incorrect classifications
def plot_image_prediction(img, label, predicted, prob_wrong, prob_right):
    plt.imshow(img, cmap="gray")
    plt.title(
        f"+{label} ({prob_right * 100:.2f}%)\n-{predicted} ({prob_wrong * 100:.2f}%)",
        fontdict={"fontsize": 9},
    )
    plt.axis(False)


def plot_predictions(label, predicted, predictions):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([0, 1])
    plot = plt.bar(range(10), predictions, color="black")
    plt.ylim([0, 1])

    plot[predicted].set_color("red")
    plot[label].set_color("blue")
