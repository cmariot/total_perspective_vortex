import numpy as np
from sklearn.model_selection import ShuffleSplit


if __name__ == "__main__":

    x = np.array([[0, 1], [2, 3], [4, 5]])
    y = np.array([0, 1, 2])

    NB_ITERATIONS = 5
    TEST_PROPORTION = 0.33

    shuffle_split = ShuffleSplit(
        n_splits=NB_ITERATIONS,
        test_size=TEST_PROPORTION
    )

    for i, (train_index, test_index) in enumerate(shuffle_split.split(x)):

        print(f"iteration {i}:")
        print(f"train_index: {train_index} \ntest_index: {test_index}")

        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f"x_train: {X_train} \ny_train: {y_train}")
        print(f"x_test: {X_test} \ny_test: {y_test}")

        print()
