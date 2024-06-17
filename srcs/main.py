# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/10 10:19:26 by cmariot           #+#    #+#              #
#    Updated: 2024/06/17 17:05:22 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from mne.io import BaseRaw
from mne.time_frequency import Spectrum
from mne import Epochs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage
from utils.parse_arguments import parse_arguments
from mne.datasets import eegbci
from preprocessing.open import open_subject_record
from preprocessing.filter import filter_data
from preprocessing.plot import plot


def preprocessing(subject_id: int, record_id: int, display_plot: bool) -> None:

    # Load and visualize the raw data
    raw: BaseRaw = open_subject_record(subject_id, record_id)

    # import mne
    # print(mne.channels.get_builtin_montages())
    # exit()
    # eegbci.standardize(raw)
    # montage = make_standard_montage("standard_1005")
    # raw.set_montage(montage)
    # raw.set_eeg_reference(projection=True)

    if display_plot:
        plot(raw, subject_id, record_id)

    # Apply a filter to the data (remove noise)
    LOW_FREQ = 8
    HIGH_FREQ = 32
    filtered: BaseRaw = filter_data(raw, low_freq=LOW_FREQ, high_freq=HIGH_FREQ)
    if display_plot:
        plot(filtered, subject_id, record_id, is_filtered=True)

    # Compute the power spectral density (PSD) of the data
    spectrum: Spectrum = filtered.compute_psd(
        method="welch",
        fmin=0.0,
        fmax=80.0,
        verbose=False
    )
    if display_plot:
        spectrum.plot(
            average=True,
            amplitude=False,
            dB=True,
        )
        plt.show()

    raw.close()

    return filtered


if __name__ == "__main__":

    try:

        subject_id, record_id = parse_arguments()
        display_plot = True

        raw: BaseRaw = preprocessing(subject_id, record_id, display_plot)

        tmin, tmax = 0., 100.0
        epochs = Epochs(
            raw,
            event_id=['T0: rest', 'T1: left fist', 'T2: right fist'],
            tmin=tmin,
            tmax=tmax,
            proj=True,
            baseline=None,
            preload=True,
            verbose=False
        )

        labels = epochs.events[:, -1] - 2

        tmin = epochs.tmin
        tmax = epochs.tmax

        epochs_train = epochs.copy().crop(tmin=tmin, tmax=tmax)

        # Define a monte-carlo cross-validation generator (reduce variance):
        scores = []
        epochs_data = epochs.get_data(copy=False)
        epochs_data_train = epochs_train.get_data(copy=False)
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        cv_split = cv.split(epochs_data_train)

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

        sfreq = raw.info["sfreq"]
        w_length = int(sfreq * 0.5)  # running classifier: window length
        w_step = int(sfreq * 0.1)  # running classifier: window step size
        w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

        scores_windows = []

        for train_idx, test_idx in cv_split:
            y_train, y_test = labels[train_idx], labels[test_idx]

            X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
            X_test = csp.transform(epochs_data_train[test_idx])

            # fit classifier
            lda.fit(X_train, y_train)

            # running classifier: test classifier on sliding window
            score_this_window = []
            for n in w_start:
                X_test = csp.transform(
                    epochs_data[test_idx][:, :, n:(n + w_length)]
                )
                score_this_window.append(lda.score(X_test, y_test))
            scores_windows.append(score_this_window)

        raw.close()

        w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

        plt.figure()
        plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
        plt.axvline(0, linestyle="--", color="k", label="Onset")
        plt.axhline(0.5, linestyle="-", color="k", label="Chance")
        plt.xlabel("time (s)")
        plt.ylabel("classification accuracy")
        plt.title("Classification score over time")
        plt.legend(loc="lower right")
        plt.show()

    except Exception as exception:
        print(exception)
    except KeyboardInterrupt:
        exit()
