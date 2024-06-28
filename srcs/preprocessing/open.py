# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    open.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/10 10:19:09 by cmariot           #+#    #+#              #
#    Updated: 2024/06/26 11:34:58 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import mne
from mne.io import concatenate_raws
from mne.io import BaseRaw
from mne.channels import DigMontage


def rename_annotations(raw: mne.io.BaseRaw, recording_id: int):
    """
    Set a more explicit name for the events
    """

    def get_events_legend(recording_id: int):
        """
        Get the events legend for T1 and T2 based on recording_id
        """
        t0_legend = "T0: rest"
        t1_legend = t2_legend = ""
        if recording_id > 2:
            t1_left_fist = [3, 4, 7, 8, 11, 12]
            if recording_id in t1_left_fist:
                t1_legend = "T1: left fist"
                t2_legend = "T2: right fist"
            else:
                t1_legend = "T1: both fists"
                t2_legend = "T2: both feet"
        return t0_legend, t1_legend, t2_legend

    t0_legend, t1_legend, t2_legend = get_events_legend(recording_id)
    raw.annotations.rename({"T0": t0_legend})
    if recording_id > 2:
        raw.annotations.rename({"T1": t1_legend, "T2": t2_legend})


def rename_channels(raw: mne.io.BaseRaw):
    new_names = {}
    for channel in raw.ch_names:
        updated_channel = channel.strip(".").upper()
        if updated_channel.endswith("Z"):
            updated_channel = updated_channel[:-1] + "z"
        if updated_channel.startswith("FP"):
            updated_channel = "Fp" + updated_channel[2:]
        new_names[channel] = updated_channel
    raw.rename_channels(new_names)


def open_subject_record(
    subject_id: int = 1, recording_id: int = 1
) -> tuple[BaseRaw, DigMontage]:

    # Load the signal of the EEG recording with the given subject_id and
    # recording_id
    # The format of the EEG recording is .edf (European Data Format)

    # Path to the dataset folder
    dataset_folder = "./dataset/"

    def subject_id_to_str(subject_id: int):
        """
        Convert a subject_id to a string with the format "Sxxx"
        """
        if subject_id < 1 or subject_id > 109:
            raise ValueError(f"Invalid subject_id: {subject_id}")
        return f"S{subject_id:03d}"

    def recording_id_to_str(recording_id: int):
        """
        Convert a recording_id to a string with the format "Rxx"
        """
        if recording_id < 1 or recording_id > 14:
            raise ValueError(f"Invalid recording_id: {recording_id}")
        return f"R{recording_id:02d}"

    str_subject_id = subject_id_to_str(subject_id)

    fist_fist = [3, 4, 7, 8, 11, 12]
    fist_foot = [5, 6, 9, 10, 13, 14]

    experiment_runs = fist_fist if recording_id in fist_fist else fist_foot

    raw_list = []
    for record in experiment_runs:

        str_recording_id = recording_id_to_str(record)
        dataset_file = f"{str_subject_id}/{str_subject_id}{str_recording_id}.edf"
        input_fname = dataset_folder + dataset_file

        # Loading a EEG recording from a .edf file
        # https://mne.tools/stable/generated/mne.io.read_raw_edf.html

        raw: mne.io.BaseRaw = mne.io.read_raw_edf(
            input_fname,
            preload=True,   # filtering require that the data be copied into RAM
            verbose=False,
        )

        raw_list.append(raw)

    raw = concatenate_raws(raw_list, verbose=True, preload=True)  # type: ignore

    picks = mne.pick_types(raw.info, eeg=True)
    raw.pick(picks)

    rename_annotations(raw, recording_id)
    rename_channels(raw)

    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    montage = raw.get_montage()
    if not montage:
        raise Exception("Error while setting the montage.")

    return raw, montage
