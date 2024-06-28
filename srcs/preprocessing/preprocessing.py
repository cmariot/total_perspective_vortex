# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    preprocessing.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/25 10:04:28 by cmariot           #+#    #+#              #
#    Updated: 2024/06/28 08:56:57 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from mne.io import BaseRaw
from mne.time_frequency import Spectrum
from mne.channels import DigMontage
import matplotlib.pyplot as plt

from srcs.preprocessing.open import open_subject_record
from srcs.preprocessing.filter import filter_data
from srcs.preprocessing.plot import plot


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


# def independent_component_analysis(raw: BaseRaw):
#     from mne.preprocessing import ICA
#     ica_raw = raw.copy()
#     n_components = 32
#     ica = ICA(n_components=n_components, random_state=42, max_iter="auto")
#     ica.fit(ica_raw)
#     exclude, scores = ica.find_bads_eog(ica_raw, ch_name='Fpz')
#     ica.exclude = exclude
#     ica_raw = ica.apply(ica_raw)
#     return ica_raw


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

    return filtered
