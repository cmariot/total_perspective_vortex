# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/24 18:49:24 by cmariot           #+#    #+#              #
#    Updated: 2024/09/27 11:49:07 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from mne import Epochs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.model_selection import ShuffleSplit, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from mne.io import BaseRaw

from mne import events_from_annotations
from sklearn.pipeline import make_pipeline


def raw_to_epochs(raw: BaseRaw):

    # Events from annotations
    events, event_id = events_from_annotations(raw)
    return Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        preload=True,
        verbose=False,
        tmin=0,
        tmax=1 / raw.info['sfreq'] * events[1][0],
        baseline=None,
        detrend=1,
        event_repeated='merge',
    )


def plot_results(w_times, scores_windows):

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
    plt.axvline(0, linestyle="--", color="k", label="Onset")
    plt.axhline(0.5, linestyle="-", color="k", label="Chance")
    plt.xlabel("time (s)")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    plt.show()


def train(raw: BaseRaw):

    epochs = raw_to_epochs(raw)

    x = epochs.get_data(copy=False)
    y = epochs.events[:, -1]

    # Cross validation
    NB_ITERATIONS, TEST_PROPORTION = 10, 0.2
    shuffle_split = ShuffleSplit(
        NB_ITERATIONS, test_size=TEST_PROPORTION, random_state=42
    )

    # Treatment pipeline
    csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    pipe = make_pipeline(csp, lda)

    # Training
    scores = cross_val_score(pipe, x, y, cv=shuffle_split)
    mean_score = np.mean(scores)
    print(f"{scores=}\n{mean_score=}\n")

    sfreq = raw.info["sfreq"]
    w_length = int(sfreq * 0.5)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, x.shape[2] - w_length, w_step)

    scores_windows = []
    cv_split = shuffle_split.split(x)
    for train_idx, test_idx in cv_split:

        x_train, y_train = (
            csp.fit_transform(x[train_idx], y[train_idx]), y[train_idx]
        )
        x_test, y_test = csp.transform(x[test_idx]), y[test_idx]

        # fit classifier
        lda.fit(x_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            x_test = csp.transform(x[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(lda.score(x_test, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

    plot_results(w_times, scores_windows)
