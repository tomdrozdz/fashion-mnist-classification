import requests
import os

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
    import numpy as np

    images_path = os.path.join(data_path, kind + suffixes[0])
    labels_path = os.path.join(data_path, kind + suffixes[1])

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 28, 28
        )

    return images, labels


def load_all_data():
    return load_data("train"), load_data("t10k")


# Normalization 0-1
def prepare_data(x, shape=None):
    if shape is not None and shape != x.shape:
        x = x.reshape(shape)

    return x / 255


def augument_data(x, y):
    return x, y
    # datagen = ImageDataGenerator(
    # rotation_range=10,
    # zoom_range=0.1,
    # width_shift_range=0.1,
    # height_shift_range=0.1
    # )


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
