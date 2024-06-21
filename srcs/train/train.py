from mne import Epochs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from mne.io import BaseRaw
from mne import events_from_annotations
from sklearn.pipeline import Pipeline
import contextlib
import io


def train(raw: BaseRaw):

    # Events from annotations
    events, event_id = events_from_annotations(raw)

    epochs = Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        preload=True,
        verbose=False,
        tmin=0,
        tmax=5,
        baseline=None,
        detrend=1,
        event_repeated='merge',
    )

    # with contextlib.redirect_stdout(io.StringIO()):
    #     epochs.plot(
    #         show=True, block=True, events=True, event_id=event_id,
    #         scalings=dict(eeg=20e-5)
    #     )
    #     plt.show()

    x = epochs.get_data(copy=False)
    y = epochs.events[:, -1]

    # Cross validation
    NB_ITERATIONS, TEST_PROPORTION = 5, 0.2
    shuffle_split = ShuffleSplit(NB_ITERATIONS, test_size=TEST_PROPORTION)

    # Dimensionality reduction algorithm : Common Spatial Patterns (CSP)
    csp = CSP(
        n_components=3,
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

    train_scores = []
    test_scores = []

    for train_idx, test_idx in shuffle_split.split(x):

        # Training
        x_train, y_train = x[train_idx], y[train_idx]
        pipe = pipe.fit(x_train, y_train)
        train_scores.append(pipe.score(x_train, y_train))
        print(f"Train score: {train_scores[-1]}\n")

        # Testing
        x_test, y_test = x[test_idx], y[test_idx]
        test_scores.append(pipe.score(x_test, y_test))
        print(f"Test score: {test_scores[-1]}\n")

    print(f"Mean train score: {np.mean(train_scores)}")
    print(f"Mean test score: {np.mean(test_scores)}")

    plt.figure()
    plt.plot(range(len(train_scores)), train_scores, label="Train score")
    plt.plot(range(len(test_scores)), test_scores, label="Test score")
    plt.xlabel("iteration")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    plt.show()
