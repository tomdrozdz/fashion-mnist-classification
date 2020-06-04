import numpy as np

M = 10


def manhattan_distance(x_test, x_train):
    return np.array([np.linalg.norm((x_train - row), axis=1, ord=1) for row in x_test])


def sort_train_labels_knn(Dist, y):
    return y[Dist.argsort()]


def p_y_x_knn(y, k):
    return np.array([np.bincount(row, minlength=M) for row in y[:, 0:k]])


def classification_accuracy(p_y_x, y_true):
    max_idxs = p_y_x.argmax(axis=1)
    accuracy = np.count_nonzero(max_idxs == y_true)

    return accuracy / len(y_true)


def model_accuracy(x_train, y_train, x_test, y_test, k=5):
    distance = manhattan_distance(x_test, x_train)
    neighbours = sort_train_labels_knn(distance, y_train)
    probability = p_y_x_knn(neighbours, k)
    accuracy = classification_accuracy(probability, y_test)

    return accuracy


if __name__ == "__main__":
    from data import get_all_data, load_all_data

    get_all_data()
    (x_train, y_train), (x_test, y_test) = load_all_data()

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Normalization 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    k = 5

    print("Running the k-nn algorithm...")
    accuracy = model_accuracy(x_train, y_train, x_test, y_test, k)

    print(f"Accuracy for knn classifier (k={k}): {accuracy}")
