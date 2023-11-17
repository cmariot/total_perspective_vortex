# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/11/17 15:21:35 by cmariot           #+#    #+#              #
#    Updated: 2023/11/17 19:11:36 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import mne


def read_raw_data():

    # Loading a EEG recording from a .edf file
    dataset_folder = "../../dataset/"
    dataset_file = "S001/S001R14.edf"
    raw = mne.io.read_raw_edf(
        input_fname=dataset_folder + dataset_file
    )

    # Print the raw data and its info
    print(raw)
    print(raw.info)

    # Plot the Power Spectral Density (PSD) for each sensor
    psd = raw.compute_psd()

    psd.plot(
        show=True
    )

    # Plot the raw sensor traces
    raw.plot(
        scalings='auto',
        title='Data from arrays',
        show=True,
        block=True
    )

    # set up and fit the ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)
    ica.plot_properties(raw)


def main():
    read_raw_data()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
