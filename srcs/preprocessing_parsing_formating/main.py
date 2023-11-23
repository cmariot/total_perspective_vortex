# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/11/17 15:21:35 by cmariot           #+#    #+#              #
#    Updated: 2023/11/22 17:20:15 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import mne
# import matplotlib.pyplot as plt


def visualize_raw_data():

    # Loading a EEG recording from a .edf file
    # https://mne.tools/stable/generated/mne.io.read_raw_edf.html

    # Path to the EDF file
    dataset_folder = "../../dataset/"
    dataset_file = "S001/S001R14.edf"
    input_fname = dataset_folder + dataset_file

    raw = mne.io.read_raw_edf(
        input_fname,
        preload=True
    )

    # channel_types = raw.get_channel_types()
    # print(f"{channel_types = }")
    # -> The 64 channels are EEG

    description = raw.describe()
    print(f"{description = }")

    # Print the raw data and its info
    print(raw)
    print(raw.info)

    # # Plot the raw sensor traces
    raw.plot(
        n_channels=10,
        scalings='auto',
        title='Data from arrays',
        show=True,
        block=True
    )

    return raw


def filter_data(raw):

    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

    ica = mne.preprocessing.ICA(
        n_components=10,
        max_iter="auto",
        random_state=97
    )

    ica.fit(filt_raw)

    explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f"Fraction of {channel_type} variance explained " +
            f"by all components: " f"{ratio}"
        )

    raw.load_data()
    # ica.plot_sources(raw, show_scrollbars=False)

    # ica.plot_components()

    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    explained_var_ratio = ica.get_explained_variance_ratio(
        filt_raw, components=[0], ch_type="eeg"
    )

    # This time, print as percentage.
    ratio_percent = round(100 * explained_var_ratio["eeg"])
    print(
        f"Fraction of variance in EEG signal explained by first component: "
        f"{ratio_percent}%"
    )

    raw.load_data()

    ica.plot_sources(
        raw,
        show_scrollbars=False,
        show=True,
        block=True
    )

    # raw.plot(
    #   order=artifact_picks,
    #   n_channels=len(artifact_picks),
    #   show_scrollbars=False
    # )

    # reconst_raw.plot(
    #     order=artifact_picks,
    #     n_channels=len(artifact_picks),
    #     show_scrollbars=False
    # )
    # del reconst_raw

    return ica


def visualize_filtered_data(filtered):
    pass


def main():
    raw = visualize_raw_data()
    filtered = filter_data(raw)
    visualize_filtered_data(filtered)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
