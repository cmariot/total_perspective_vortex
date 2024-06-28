import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws
from mne.time_frequency import Spectrum
from preprocessing.open import open_subject_record


if __name__ == "__main__":

    # Tutorial :
    # https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html

    subject = 3
    runs = [3, 4, 7, 8, 11, 12]

    raw = concatenate_raws([open_subject_record(subject, run)[0] for run in runs])

    eegbci.standardize(raw)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)

    filtered = raw.filter(
        8.0, 40.0, fir_design="firwin", skip_by_annotation="edge"
    )

    # Compute the power spectral density (PSD) of the data
    spectrum: Spectrum = filtered.compute_psd(
        method="welch",
        fmin=0,
        fmax=80.0,
    )
    spectrum.plot(
        average=True,
        amplitude=False,
        dB=True,
    )
    plt.show()

    tmin, tmax = 0, 5.0
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    epochs = Epochs(
        raw,
        event_id=['T0: rest', 'T1: left fist', 'T2: right fist'],
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )

    epochs_train = epochs.copy()
    labels = epochs.events[:, -1] - 2

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data(copy=False)
    epochs_data_train = epochs_train.get_data(copy=False)
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1.0 - class_balance)

    print("\n" + "=" * 80 + "\n")
    print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")
    print("\n" + "=" * 80 + "\n")

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)
    plt.show()

    # ************************ #
    # Classification over time #
    # ************************ #

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
            X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
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
