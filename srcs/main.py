# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/10 10:19:26 by cmariot           #+#    #+#              #
#    Updated: 2024/06/10 18:03:15 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import mne
from preprocessing_parsing_formating.open import open_raw_data
from preprocessing_parsing_formating.filter import filter_data
from preprocessing_parsing_formating.plot import plot


if __name__ == "__main__":
    try:
        for patient_id in range(1, 110):
            for recording_id in range(1, 15):

                # patient_id = 1
                # recording_id = 4

                if patient_id == 100 and recording_id > 2:
                    # Patient 100 Fc5. channel is not connected ?
                    # RuntimeWarning: Limited 1 annotation(s) that were
                    # expanding outside the data range.
                    # continue
                    exit()

                print(f"Patient ID: {patient_id} - Recording ID: {recording_id}")

                # Load and visualize the raw data
                raw: mne.io.BaseRaw = open_raw_data(patient_id, recording_id)
                # raw = plot(raw, patient_id, recording_id)

                # Apply a filter to the data (remove noise)
                filtered: mne.io.BaseRaw = filter_data(
                    raw, low_freq=8.0, high_freq=32.0
                )
                # filtered = plot(
                #     filtered, patient_id, recording_id, is_filtered=True
                # )

                # Compute the power spectral density (PSD) of the data
                spectrum = filtered.compute_psd()
                spectrum.plot(
                    average=True,
                    amplitude=False,
                    dB=False,
                    show=True
                )
                import matplotlib.pyplot as plt
                plt.show()

                raw.close()
                filtered.close()

    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        exit()
