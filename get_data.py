import requests
import os.path

url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

data = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def get_data(path="data"):
    if not os.path.exists(path):
        os.makedirs(path)

    print("Checking if data is downloaded:")

    for file in data:
        current_path = os.path.join(path, file)
        
        if os.path.isfile(current_path):
            print(f"\t{file} already downloaded")
        else:
            print(f"\tDownloading {file}...")
            r = requests.get(url + file)
            with open(current_path, "wb") as f:
                f.write(r.content)


if __name__ == "__main__":
    get_data()
