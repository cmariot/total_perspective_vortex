# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/10 10:19:26 by cmariot           #+#    #+#              #
#    Updated: 2024/06/17 10:25:50 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import mne
import matplotlib.pyplot as plt


from utils.parse_arguments import parse_arguments
# from utils.print_title import print_title

from preprocessing.open import open_subject_record
from preprocessing.filter import filter_data
from preprocessing.plot import plot


if __name__ == "__main__":

    try:

        subject_id, record_id = parse_arguments()

        # Load and visualize the raw data
        raw: mne.io.BaseRaw = open_subject_record(subject_id, record_id)
        raw = plot(raw, subject_id, record_id)

        LOW_FREQ = 10
        HIGH_FREQ = 40

        # Apply a filter to the data (remove noise)
        filtered: mne.io.BaseRaw = filter_data(
            raw, low_freq=LOW_FREQ, high_freq=HIGH_FREQ
        )
        filtered = plot(filtered, subject_id, record_id, is_filtered=True)

        # print(filtered)
        # print(filtered.info)

        # Compute the power spectral density (PSD) of the data
        spectrum: mne.time_frequency.Spectrum = filtered.compute_psd(
            method="welch",
            fmin=0,
            fmax=80.0,
            verbose=False
        )
        spectrum.plot(
            average=True,
            amplitude=False,
            dB=True,
        )
        plt.show()

        # events
        # events, event_dict = mne.events_from_annotations(
        #     filtered, verbose=False
        # )

        # # print(events)

        # epochs = mne.Epochs(
        #     filtered,
        #     events,
        #     event_dict,
        #     baseline=None,
        # )

        # print(epochs)
        # Number of features = number of electrodes (64)

        # Apply a Dimensionality reduction algorithm
        # Principal component analysis (PCA) : transform potentially
        # correlated variables into a smaller set of variables, called
        # principal components

        # x.shape = (64, 1025)

        # x = dimensionality_reduction(
        #     x=spectrum.get_data()
        # )


        # calibration phase

        TIMEFRAME = 1   # 1 second
        record_len = raw.info
        print(record_len)

        raw.close()
        filtered.close()

    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        exit()
