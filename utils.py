import matplotlib.pyplot as plt
import numpy as np

# Utility for displaying the images
def show_image(img, label):
    plt.imshow(img, cmap="gray")
    plt.title(label)
    plt.axis(False)
    plt.show()


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
