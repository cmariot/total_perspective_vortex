try:

    import mne
    import matplotlib.pyplot as plt
    from mne.datasets import eegbci
    from mne import Epochs
    from mne import events_from_annotations
    import numpy as np
    from mne.io import read_raw_edf, concatenate_raws
    from mne import pick_types
    # from .helper import freq_selection_class_dis
    # from sklearn.model_selection import ShuffleSplit

except ImportError:
    raise ImportError(
        "Please the required dependencies" +
        " by running: pip install -r requirements.txt"
    )


# https://mne.tools/dev/auto_examples/decoding/decoding_csp_eeg.html#sphx-glr-auto-examples-decoding-decoding-csp-eeg-py
# https://pyriemann.readthedocs.io/en/latest/auto_examples/motor-imagery/plot_single.html
# https://github.com/JGalego/eeg-bci-tutorial/blob/master/eeg_bci.ipynb


def load_dataset_files(subject, task):
    raw_fnames = eegbci.load_data(subject, task["experiment"])
    raw: mne.io.BaseRaw = concatenate_raws(
        [read_raw_edf(f, preload=True) for f in raw_fnames]
    )
    return raw


def pick_channels(raw: mne.io.BaseRaw):
    picks = pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
    )
    return picks


def rename_channels(raw: mne.io.BaseRaw):
    eegbci.standardize(raw)


def rename_annotations(raw: mne.io.BaseRaw, task: dict):
    """
    Rename the annotations of the task
    """
    if task["id"] < 3:
        annotations = {"T0": "Rest"}
    else:
        annotations = {
            "T0": "Rest",
            "T1": "Left fist" if task["id"] in (3, 4) else "Both fists",
            "T2": "Right fist" if task["id"] in (3, 4) else "Both feet"
        }
    raw.annotations.rename({"T0": "Rest"})
    if "T1" in annotations:
        raw.annotations.rename({"T1": annotations["T1"]})
    if "T2" in annotations:
        raw.annotations.rename({"T2": annotations["T2"]})
    # Drop T0 annotations
    idx = np.where(raw.annotations.description == "Rest")
    if len(idx[0]) > 0:
        raw.annotations.delete(idx)
    # Drop annotations that are not in the task annotations
    idx = np.where(
        (raw.annotations.description != annotations["T1"]) &
        (raw.annotations.description != annotations["T2"])
    )
    if len(idx[0]) > 0:
        raw.annotations.delete(idx)


first_experiment = True


def set_montage(raw: mne.io.BaseRaw, display_plot):
    """
    Set the montage of the raw data
    """

    global first_experiment

    raw.set_montage('standard_1005')
    montage = raw.get_montage()
    if not montage:
        raise Exception("Error while setting the montage.")

    if display_plot and first_experiment:

        # Plot electrodes position (2D)
        montage.plot(
            verbose=False,
            sphere=((0.0, 0.016, 0.0, 0.095))
        )
        plt.show()

        # Plot electrodes position (3D)
        montage.plot(
            kind="3d",
            show=True,
            verbose=False,
            sphere=((0.0, 0.016, 0.0, 0.095))
        )
        plt.show()

    return montage


def plot_raw_data(raw: mne.io.BaseRaw, display_plot, subject, task):

    if display_plot:

        raw.plot(
            duration=15,
            n_channels=10,
            scalings=dict(eeg=1e-4),
            title=f"Subject {subject:03d} - Task {task['name']} - Raw",
            show=True,
            block=True,
            verbose=False
        )
        plt.show()


def apply_filters(raw: mne.io.BaseRaw, picks):

    # print("Apply filters")

    # cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # print("cv ok")

    # events, event_id = events_from_annotations(raw)
    # print("events_from_annotations ok")

    # tmin, tmax = 0.5, 2.5

    # freq_band = [5., 35.]
    # sub_band_width = 4.
    # sub_band_step = 2.
    # alpha = 0.4

    # # Select frequency band using training set
    # best_freq, all_class_dis = \
    #     freq_selection_class_dis(
    #         raw, freq_band, sub_band_width,
    #         sub_band_step, alpha,
    #         tmin, tmax,
    #         picks, event_id,
    #         cv,
    #         return_class_dis=True, verbose=False
    #     )
    # print("freq_selection_class_dis ok")

    # print(
    #     'Selected frequency band : {} - {} Hz'.format(
    #         best_freq[0][0], best_freq[0][1]
    #     )
    # )

    # exit()

    LOW_FREQ = 7.0
    HIGH_FREQ = 35.0

    raw.filter(
        LOW_FREQ, HIGH_FREQ,
        fir_design='firwin',
        method='iir',
        picks=picks
    )


def plot_filtered_data(raw: mne.io.BaseRaw, display_plot, subject, task):

    global first_experiment

    if display_plot:
        raw.plot(
            duration=15,
            n_channels=10,
            scalings=dict(eeg=1e-4),
            title=f"Subject {subject:03d} - Task {task['name']} - Filtered",
            show=True,
            block=True,
            verbose=False
        )
        plt.show()
        if first_experiment:
            raw.compute_psd().plot(
                average=False,
                show=True,
                sphere=((0.0, 0.016, 0.0, 0.095)),
            )
            plt.show()
            raw.compute_psd().plot(
                average=True,
                show=True,
            )
            plt.show()
            first_experiment = False


def get_epochs(raw: mne.io.BaseRaw):
    tmin, tmax = 0.0, 4.
    events, event_id = events_from_annotations(raw)
    epochs = Epochs(
        raw, events, event_id,
        tmin, tmax,
        baseline=None, preload=True
    )
    return epochs


def preprocessing(subject: int, task: dict, mode: str):

    display_plot = mode == "preprocessing"
    mne.set_log_level(verbose="CRITICAL")

    raw = load_dataset_files(subject, task)
    picks = pick_channels(raw)
    rename_channels(raw)
    rename_annotations(raw, task)
    set_montage(raw, display_plot)
    plot_raw_data(raw, display_plot, subject, task)
    apply_filters(raw, picks)
    plot_filtered_data(raw, display_plot, subject, task)
    epochs = get_epochs(raw)

    return epochs
