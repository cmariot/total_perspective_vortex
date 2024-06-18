from mne import Epochs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.model_selection import ShuffleSplit
# import numpy as np
# import matplotlib.pyplot as plt
from mne.io import BaseRaw
from mne import events_from_annotations
from sklearn.pipeline import Pipeline


def train(raw: BaseRaw):

    # Events from annotations
    events, event_id = events_from_annotations(
        raw
    )

    epochs = Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        preload=True,
        verbose=False,
    )

    # epochs.plot(
    #     scalings=dict(eeg=1e-5),
    #     show=True,
    #     block=True,
    # )
    # plt.show()

    data = epochs.get_data()
    labels = epochs.events[:, -1]

    data_splitted = ShuffleSplit(n_splits=10, test_size=0.2).split(data)

    # Dimensionality reduction algorithm : Common Spatial Patterns (CSP)
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    estimators = [
        ("CSP", csp),
        ("LDA", lda)
    ]
    pipe = Pipeline(estimators)


    # sfreq = raw.info["sfreq"]
    # w_length = int(sfreq * 0.5)  # running classifier: window length
    # w_step = int(sfreq * 0.1)  # running classifier: window step size
    # w_start = np.arange(0, data.shape[2] - w_length, w_step)
    scores = []
    for i, (train_idx, test_idx) in enumerate(data_splitted):
        x_train, y_train = data[train_idx], labels[train_idx]
        pipe.fit(x_train, y_train)
        continue

        print(f"{y_train.shape=}, {y_test.shape=}")
        print(y_train)
        print(y_test)
        continue

        # print(f"{X_train.shape=}, {y_train.shape=}")
        # print(f"{X_test.shape=}, {y_test.shape=}")

        # fit classifier
        lda.fit(X_train, y_train)

        # # running classifier: test classifier on sliding window
        # test_score = []
        # for n in w_start:
        #     X_test = csp.transform(
        #         data[test_idx][:, :, n:(n + w_length)]
        #     )
        #     test_score.append(lda.score(X_test, y_test))
        # test_score.append(test_score)

        print("")

    # w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

    # plt.figure()
    # plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
    # plt.axvline(0, linestyle="--", color="k", label="Onset")
    # plt.axhline(0.5, linestyle="-", color="k", label="Chance")
    # plt.xlabel("time (s)")
    # plt.ylabel("classification accuracy")
    # plt.title("Classification score over time")
    # plt.legend(loc="lower right")
    # plt.show()
