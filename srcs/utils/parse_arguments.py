import argparse


def parse_arguments():

    parser = argparse.ArgumentParser(
        prog='Total Perspective Vortex',
        description='Brain computer interface with machine learning based on ' +
                    'electroencephalographic data',
        epilog='Text at the bottom of help'
    )

    parser.add_argument(
        '-s', '--subject_id',
        type=int,
        default=32,
        help='Subject ID',
    )

    parser.add_argument(
        '-r', '--record_id',
        type=int,
        default=8,
        help='ID of the recording session',

    )

    args = parser.parse_args()

    if args.subject_id == 100 and args.record_id > 2:
        # Subject 100 Fc5. channel is not connected ?
        # RuntimeWarning: Limited 1 annotation(s) that were
        # expanding outside the data range.
        # continue
        exit()

    return (
        args.subject_id,
        args.record_id
    )
