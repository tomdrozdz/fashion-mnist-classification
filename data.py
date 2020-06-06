import requests
import os
import numpy as np


url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

suffixes = [
    "-images-idx3-ubyte.gz",
    "-labels-idx1-ubyte.gz",
]

data_path = "data"


def get_data(kind="t10k"):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    print(f"Checking if {kind} data is downloaded:")

    for suffix in suffixes:
        file_name = kind + suffix
        current_path = os.path.join(data_path, file_name)

        if os.path.isfile(current_path):
            print(f"\t{file_name} already downloaded")
        else:
            print(f"\tDownloading {file_name}...")
            r = requests.get(url + file_name)
            with open(current_path, "wb") as f:
                f.write(r.content)


def get_all_data():
    get_data("train")
    get_data("t10k")


def load_data(kind="t10k"):
    import gzip

    images_path = os.path.join(data_path, kind + suffixes[0])
    labels_path = os.path.join(data_path, kind + suffixes[1])

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 28, 28, 1
        )

    return images, labels


def load_all_data():
    return load_data("train"), load_data("t10k")


# Normalization 0-1
def prepare_data(x, shape=None):
    if shape and shape != x.shape:
        x = x.reshape(shape)

    return x / 255


def split_data(x, y, split=0.15):
    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split)
    return x_train, x_val, y_train, y_val


def augument_data(x, y, erasion_prob=0.5, mul=3, shape=(28, 28, 1)):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from erasing import get_random_eraser

    print("Augumenting data...")

    eraser = get_random_eraser(p=erasion_prob)

    datagen = ImageDataGenerator(
        #rotation_range=10,
        horizontal_flip=True,
        preprocessing_function=eraser,
        zoom_range=1.1,
    )

    n = mul - 1

    x_new = []
    y_new = []

    def gen_images():
        for img in x:
            img = img.reshape((1, *shape))
            gen = datagen.flow(img, batch_size=1)
            for i in range(n):
                yield gen.next().reshape(shape)

    x_new = np.array([img for img in gen_images()])

    x = np.append(x, x_new, axis=0)
    y = np.append(y, np.repeat(y, n))

    return x, y


def get_labels():
    return [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]


if __name__ == "__main__":
    get_all_data()
