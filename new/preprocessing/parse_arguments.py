# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    parse_arguments.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/28 13:39:21 by cmariot           #+#    #+#              #
#    Updated: 2024/09/30 09:45:33 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


try:
    import pyfiglet
    import argparse
except ImportError:
    raise ImportError(
        "Please the required dependencies" +
        " by running: pip install -r requirements.txt"
    )


def header():

    """
    Print the Total Perspective Vortex header
    """

    pyfiglet.print_figlet(
        "Brain Computer Interface",
        font="small", colors="cyan", width=100
    )


def get_command_line_arguments() -> tuple:

    """
    Get the command line arguments with argparse
    Returns:
        subjects: list of subjects ID
        experiments: list of experiments ID
        mode: program mode
    """

    parser = argparse.ArgumentParser(
        prog='Total Perspective Vortex',
        description='Brain computer interface with machine learning based on ' +
                    'electroencephalographic data',
    )

    # List of subjects
    parser.add_argument(
        '-s', '--subjects',
        type=int,
        nargs='*',
        default=list(range(1, 110)),
        help='Subjects ID'
    )

    # List of experiments
    parser.add_argument(
        '-e', '--experiments',
        type=int,
        nargs='*',
        default=list(range(1, 15)),
        help='Experiments ID'
    )

    # Program mode (preprocessing, train, predict)
    parser.add_argument(
        type=str,
        default='train',
        help='Program mode',
        action='store',
        choices=['preprocessing', 'train', 'predict'],
        metavar='MODE',
        dest='mode',
    )

    args = parser.parse_args()

    return (args.subjects, args.experiments, args.mode)


def check_arguments(subjects: list[int], experiments: list[int], mode: str):

    """
    Check if all the arguments are provided and valid
    """

    if not subjects:
        raise ValueError("Please provide a valid subject ID")
    elif not experiments:
        raise ValueError("Please provide a valid experiment ID")
    elif mode not in ['preprocessing', 'train', 'predict']:
        raise ValueError("Please provide a valid mode")

    if not all(isinstance(subject, int) for subject in subjects):
        raise ValueError("Please provide valid subject IDs")
    elif not all(isinstance(experiment, int) for experiment in experiments):
        raise ValueError("Please provide valid experiment IDs")
    elif not isinstance(mode, str):
        raise ValueError("Please provide a valid mode")

    for subject in subjects:
        if subject < 1 or subject > 109:
            raise ValueError("Please provide a valid subject ID")
    for experiment in experiments:
        if experiment < 1 or experiment > 14:
            raise ValueError("Please provide a valid experiment ID")


def get_selected_tasks(experiments):

    # =========  ===================================
    # run        task
    # =========  ===================================
    # 1          Baseline, eyes open
    # 2          Baseline, eyes closed
    # 3, 7, 11   Motor execution: left vs right hand
    # 4, 8, 12   Motor imagery: left vs right hand
    # 5, 9, 13   Motor execution: hands vs feet
    # 6, 10, 14  Motor imagery: hands vs feet
    # =========  ===================================

    experimental_tasks = (
        # {"id": 1,
        #  "name": "1: Baseline, eyes open",
        #  "experiment": (1,)},
        # {"id": 2,
        #  "name": "2: Baseline, eyes closed",
        #  "experiment": (2,)},
        {"id": 3,
         "name": "1: Open and close left or right fist",
         "experiment": (3, 7, 11)},
        {"id": 4,
         "name": "2: Imagine opening and closing left or right fist",
         "experiment": (4, 8, 12)},
        {"id": 5,
         "name": "3: Open and close both fists or both feet",
         "experiment": (5, 9, 13)},
        {"id": 6,
         "name": "4: Imagine opening and closing both fists or both feet",
         "experiment": (6, 10, 14)}
    )

    # Keep only the experiments that the user wants to process
    selected_tasks = {}
    for experiments in experiments:
        for task in experimental_tasks:
            if experiments in task["experiment"]:
                if task["id"] not in selected_tasks:
                    selected_tasks[task["id"]] = {
                        "id": task["id"],
                        "name": task["name"],
                        "experiment": [experiments]
                    }
                else:
                    selected_tasks[task["id"]]["experiment"].append(experiments)
    selected_tasks = tuple(selected_tasks.values())

    return selected_tasks


def parse_arguments():

    header()
    subjects, experiments, mode = get_command_line_arguments()
    check_arguments(subjects, experiments, mode)
    selected_tasks = get_selected_tasks(experiments)
    return subjects, selected_tasks, mode
