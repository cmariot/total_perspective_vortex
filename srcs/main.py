# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/10 10:19:26 by cmariot           #+#    #+#              #
#    Updated: 2024/06/11 14:06:07 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import mne
import matplotlib.pyplot as plt

from preprocessing_parsing_formating.open import open_raw_data
from preprocessing_parsing_formating.filter import filter_data
from preprocessing_parsing_formating.plot import plot
from preprocessing_parsing_formating.dimensionality_reduction import dimensionality_reduction


def print_title(title: str):
    """
    Print a title
    """
    BLUE = "\033[94m"
    END = "\033[0m"

    print("\n" + "=" * 80)
    print(BLUE + title + END)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        for patient_id in range(1, 110):
            for recording_id in range(1, 15):

                if patient_id == 100 and recording_id > 2:
                    # Patient 100 Fc5. channel is not connected ?
                    # RuntimeWarning: Limited 1 annotation(s) that were
                    # expanding outside the data range.
                    # continue
                    exit()

                print(f"Patient ID: {patient_id} - Recording ID: {recording_id}")

                # Load and visualize the raw data
                print_title("Loading and visualizing the raw data")
                raw: mne.io.BaseRaw = open_raw_data(patient_id, recording_id)
                # raw = plot(raw, patient_id, recording_id)

                # Apply a filter to the data (remove noise)
                filtered: mne.io.BaseRaw = filter_data(
                    raw, low_freq=8.0, high_freq=32.0
                )
                # filtered = raw
                # filtered = plot(
                #     filtered, patient_id, recording_id, is_filtered=True
                # )

                print(filtered)
                print(filtered.info)

                # Compute the power spectral density (PSD) of the data
                spectrum: mne.time_frequency.Spectrum = filtered.compute_psd(
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

                # events
                events, event_dict = mne.events_from_annotations(
                    filtered, verbose=False
                )

                print(events)

                epochs = mne.Epochs(
                    filtered,
                    events,
                    event_dict,
                    baseline=None,
                )

                print(epochs)
                # Number of features = number of electrodes (64)

                # Apply a Dimensionality reduction algorithm
                # Principal component analysis (PCA) : transform potentially
                # correlated variables into a smaller set of variables, called
                # principal components

                # x.shape = (64, 1025)

                # x = dimensionality_reduction(
                #     x=spectrum.get_data()
                # )

                raw.close()
                filtered.close()

    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        exit()
