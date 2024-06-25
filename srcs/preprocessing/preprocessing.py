# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    preprocessing.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/25 10:04:28 by cmariot           #+#    #+#              #
#    Updated: 2024/06/25 15:05:06 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from mne.io import BaseRaw
from mne.time_frequency import Spectrum
from mne.channels import DigMontage
import matplotlib.pyplot as plt

from srcs.preprocessing.open import open_subject_record
from srcs.preprocessing.filter import filter_data
from srcs.preprocessing.plot import plot

from mne.report import Report


def visualize_data(
    raw: BaseRaw, montage: DigMontage, filtered: BaseRaw,
    spectrum: Spectrum, subject_id: int, record_id: int
):

    """
    Display the data in different ways using MNE plots
    """

    # Plot electrodes position (2D)
    montage.plot(sphere=((0.0, 0.015, 0.0, 0.095)), verbose=False)
    plt.show()

    # Plot electrodes position (3D)
    montage.plot(kind="3d", show=True, verbose=False)
    plt.show()

    # Original data
    plot(raw, subject_id, record_id)

    # PSD Scalp topography
    spectrum.plot_topomap()
    plt.show()

    # Filtered data
    plot(filtered, subject_id, record_id, is_filtered=True)

    # PSD per channel
    spectrum.plot(average=False, amplitude=False, dB=True)
    plt.show()

    # Average PSD
    spectrum.plot(average=True, amplitude=False, dB=True)
    plt.show()


def preprocessing(
    subject_id: int, record_id: int, display_plot: bool
) -> BaseRaw:

    # Load and visualize the raw data
    raw, montage = open_subject_record(subject_id, record_id)

    # Apply a filter to the data (remove noise)
    filtered: BaseRaw = filter_data(raw)

    # Compute the power spectral density (PSD) of the data
    spectrum: Spectrum = filtered.compute_psd(verbose=True)

    # Plots
    if display_plot:
        visualize_data(raw, montage, filtered, spectrum, subject_id, record_id)

    rep = Report()
    print(rep)

    # Perform ICA test
    from mne.preprocessing import ICA

    n_components = 5
    ica = ICA(
        n_components=n_components,
        random_state=42,
        max_iter="auto"
    ).fit(filtered)
    ica.exclude = [0]
    raw_ica = ica.apply(filtered)

    # Original data
    plot(filtered, subject_id, record_id)

    # ICA data
    plot(raw_ica, subject_id, record_id, is_filtered=True)

    exit()

    return filtered
