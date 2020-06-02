from get_data import get_data
from load_data import load_data

data_path = "data"

def train_network():
    get_data(data_path)
    x_train, y_train = load_data(data_path, "train")
    x_test, y_test = load_data(data_path, "t10k")
    
    print(len(x_train[1]))

if __name__ == "__main__":
    train_network()
