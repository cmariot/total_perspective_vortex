# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    filter.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/10 10:19:14 by cmariot           #+#    #+#              #
#    Updated: 2024/09/27 15:11:17 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import mne


def filter_data(raw: mne.io.BaseRaw):
    """
    The EEG (electroencephalographic) signals are often noisy and contain
    artifacts (such as eye movements or electrical interference) that can
    disrupt the analysis.
    A filter can removes or attenuates parts of a signal.
    Here, we use a band-pass filter that attenuates all frequencies outside of
    the [low_freq, high_freq] range.
    The frequencies in the 1-40 Hz range contain the relevant brain information
    for the analysis, such as the alpha (8-13 Hz) and beta (13-30 Hz) rhythms.
    """

    # Band-pass filter to keep only the alpha and beta rhythms
    LOW_FREQ = 8.0
    HIGH_FREQ = 30.0
    filtered = raw.copy()
    filtered.filter(LOW_FREQ, HIGH_FREQ, verbose=False)

    # Notch filter to remove the 60 Hz line noise
    filtered.notch_filter(60, verbose=False)

    return filtered
