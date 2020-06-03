




def train_network(x_train, y_train, x_test, y_test):
    print(len(x_train[1]))


if __name__ == "__main__":
    from data import get_all_data, load_all_data
    
    data_path = "data"
    
    get_all_data(data_path)
    (x_train, y_train), (x_test, y_test) = load_all_data(data_path)
    
    train_network(x_train, y_train, x_test, y_test)
