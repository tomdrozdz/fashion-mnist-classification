import requests
import os.path

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

if __name__ == "__main__":
    get_all_data()
