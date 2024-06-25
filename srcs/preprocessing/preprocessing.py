# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    preprocessing.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/25 10:04:28 by cmariot           #+#    #+#              #
#    Updated: 2024/06/25 10:16:26 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

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
    raw, montage = open_subject_record(subject_id, record_id)

    # Apply a filter to the data (remove noise)
    filtered: BaseRaw = filter_data(raw)

    # Compute the power spectral density (PSD) of the data
    spectrum: Spectrum = filtered.compute_psd(verbose=True)

    # Plots
    if display_plot:

        # Plot electrodes position (2D)
        montage.plot(sphere=((0.0, 0.015, 0.0, 0.095)))
        plt.show()

        # Plot electrodes position (3D)
        montage.plot(kind="3d", show=True)
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

    raw.close()

    return filtered
