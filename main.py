# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/10 10:19:26 by cmariot           #+#    #+#              #
#    Updated: 2024/06/18 12:02:53 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from mne.io import BaseRaw

from srcs.utils.parse_arguments import parse_arguments
from srcs.preprocessing.preprocessing import preprocessing
from srcs.train.train import train


if __name__ == "__main__":

    try:

        subject_id, record_id, program_mode = parse_arguments()
        display_plot = True if program_mode == 'preprocessing' else False

        print(f"Subject ID: {subject_id}")
        print(f"Recording ID: {record_id}")
        print(f"Program mode: {program_mode}")
        print(f"Display plot: {display_plot}")

        raw: BaseRaw = preprocessing(subject_id, record_id, display_plot)

        if program_mode == 'train':
            model = train(raw)

        raw.close()

    except Exception as exception:
        print(exception)
    except KeyboardInterrupt:
        exit()
