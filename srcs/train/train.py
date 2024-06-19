from mne import Epochs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from mne.io import BaseRaw
from mne import events_from_annotations
from sklearn.pipeline import Pipeline


def train(raw: BaseRaw):

    # Events from annotations
    events, event_id = events_from_annotations(raw)

    epochs = Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        preload=True,
        verbose=False,
    )

    x = epochs.get_data(copy=False)
    y = epochs.events[:, -1]

    # Cross validation
    NB_ITERATIONS, TEST_PROPORTION = 10, 0.2
    shuffle_split = ShuffleSplit(NB_ITERATIONS, test_size=TEST_PROPORTION)

    # Dimensionality reduction algorithm : Common Spatial Patterns (CSP)
    csp = CSP(
        n_components=4,
        cov_est="concat",
        cov_method_params=None,
        log=True,
        norm_trace=False,
        reg=None,
        transform_into='average_power'
    )

    # Classifier
    lda = LinearDiscriminantAnalysis(
        solver='svd',
        store_covariance=False,
        tol=0.0001
    )

    # Treatment pipeline
    pipe = Pipeline(
        steps=[
            ("CSP", csp),
            ("LDA", lda)
        ],
        memory=None,
        verbose=False
    )

    scores = []

    for train_idx, test_idx in shuffle_split.split(x):

        x_train, y_train = x[train_idx], y[train_idx]

        print("1")
        print(x_train[0])
        print(x[train_idx[0]])

        if any(x_train[0] != x[train_idx[0]]):
            print("ERROR")
            exit()
        print("1")
        if y_train[0] != y[train_idx[0]]:
            print("ERROR")
            exit()

        pipe = pipe.fit(x_train, y_train)
        x_test, y_test = x[test_idx], y[test_idx]
        scores.append(pipe.score(x_test, y_test))
        print(f"Score: {scores[-1]}\n")

    print(f"Mean score: {np.mean(scores)}")

    plt.figure()
    plt.plot(range(len(scores)), scores, label="Score")
    plt.xlabel("iteration")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    plt.show()
