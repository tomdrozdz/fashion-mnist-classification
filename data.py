import requests
import os

url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

suffixes = [
    "-images-idx3-ubyte.gz",
    "-labels-idx1-ubyte.gz",
]


def get_data(path="data", kind="t10k"):
    if not os.path.exists(path):
        os.makedirs(path)

    print(f"Checking if {kind} data is downloaded:")

    for suffix in suffixes:
        file = kind + suffix
        current_path = os.path.join(path, file)

        if os.path.isfile(current_path):
            print(f"\t{file} already downloaded")
        else:
            print(f"\tDownloading {file}...")
            r = requests.get(url + file)
            with open(current_path, "wb") as f:
                f.write(r.content)


def get_all_data(path="data"):
    get_data(path, "train")
    get_data(path, "t10k")


# Script copied from the fashion-mnist repository
# Provided by the authors of the dataset
def load_data(path="data", kind="t10k"):
    import gzip
    import numpy as np

    images_path = os.path.join(path, f"{kind + suffixes[0]}")
    labels_path = os.path.join(path, f"{kind + suffixes[1]}")

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


def load_all_data(path="data"):
    return load_data(path, "train"), load_data(path, "t10k")


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
