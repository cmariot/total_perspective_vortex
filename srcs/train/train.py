# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/24 18:49:24 by cmariot           #+#    #+#              #
#    Updated: 2024/06/24 19:27:17 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from mne import Epochs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.model_selection import ShuffleSplit
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
        tmax=1 / raw.info['sfreq'] * 50,
        baseline=None,
        detrend=1,
        event_repeated='merge',
    )


def plot_results(train_scores, test_scores):

    print(f"Mean train score: {np.mean(train_scores)}")
    print(f"Mean test score: {np.mean(test_scores)}")

    plt.figure()
    plt.plot(range(len(train_scores)), train_scores, label="Train score")
    plt.plot(range(len(test_scores)), test_scores, label="Test score")
    plt.xlabel("iteration")
    plt.ylabel("classification accuracy")
    plt.ylim(0, 1)
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    plt.show()


def train(raw: BaseRaw):

    epochs = raw_to_epochs(raw)

    x = epochs.get_data(copy=False)
    y = epochs.events[:, -1]

    # Cross validation
    NB_ITERATIONS, TEST_PROPORTION = 20, 0.3
    shuffle_split = ShuffleSplit(NB_ITERATIONS, test_size=TEST_PROPORTION)

    # Treatment pipeline
    pipe = make_pipeline(
        CSP(n_components=5),
        LinearDiscriminantAnalysis()
    )

    train_scores, test_scores = [], []
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

    plot_results(train_scores, test_scores)
