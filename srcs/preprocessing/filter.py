# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    filter.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/10 10:19:14 by cmariot           #+#    #+#              #
#    Updated: 2024/06/10 10:19:15 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import mne


def filter_data(raw: mne.io.BaseRaw, low_freq: float, high_freq: float):
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
    copy = raw.copy()
    copy.filter(low_freq, high_freq, verbose=False)
    return copy
