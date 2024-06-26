# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/09 13:47:47 by cmariot           #+#    #+#              #
#    Updated: 2024/06/18 18:20:22 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


from mne.io import BaseRaw
import matplotlib.pyplot as plt


def plot(
    raw: BaseRaw,
    subject_id: int,
    recording_id: int,
    is_filtered: bool = False
):

    """
    Plot the raw sensor traces with the events
    X-axis: time in seconds
    Y-axis: sensor channels

    The events are marked as :
    T0 corresponds to rest
    T1 corresponds to onset of motion (real or imagined)
      of the left fist (in runs 3, 4, 7, 8, 11, and 12)
      both fists (in runs 5, 6, 9, 10, 13, and 14)
    T2 corresponds to onset of motion (real or imagined)
      of the right fist (in runs 3, 4, 7, 8, 11, and 12)
      both feet (in runs 5, 6, 9, 10, 13, and 14)
    """

    def get_plot_title(subject_id: int, recording_id: int):
        """
        Get the title of the plot
        """
        t0_legend = "T0: rest"
        if not is_filtered:
            pre_title = "Raw "
        else:
            pre_title = "Filtered "
        title = pre_title + (
            f"EEG recording - Subject ID: {subject_id} " +
            f"- Recording ID: {recording_id} - {t0_legend}"
        )
        if recording_id > 2:
            t1_left_fist = [3, 4, 7, 8, 11, 12]
            if recording_id in t1_left_fist:
                t1_legend = "T1: left fist"
                t2_legend = "T2: right fist"
            else:
                t1_legend = "T1: both fists"
                t2_legend = "T2: both feet"
            title += f" - {t1_legend} - {t2_legend}"
        return title

    raw.plot(
        n_channels=10,
        scalings=dict(eeg=10e-5),
        title=get_plot_title(subject_id, recording_id),
        show=True,
        block=True,
        verbose=False
    )
    plt.show()
