from mne.io import BaseRaw
from mne.time_frequency import Spectrum
import matplotlib.pyplot as plt

from srcs.preprocessing.open import open_subject_record
from srcs.preprocessing.filter import filter_data
from srcs.preprocessing.plot import plot


def preprocessing(
    subject_id: int, record_id: int, display_plot: bool
) -> BaseRaw:

    # Load and visualize the raw data
    raw: BaseRaw = open_subject_record(subject_id, record_id)
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
        fmin=0,
        fmax=80,
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
