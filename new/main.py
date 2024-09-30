try:
    from preprocessing.parse_arguments import parse_arguments
    from preprocessing.preprocessing import preprocessing
    from train.train import train
    import numpy as np
except ImportError:
    raise ImportError(
        "Please install the required dependencies" +
        " by running: pip install -r requirements.txt"
    )


# def compute_accuracy(value):
#     accuracy = np.array([])
#     while True:
#         accuracy.append(value)
#         yield accuracy
#     return accuracy.mean()


def main():

    subjects, tasks, mode = parse_arguments()
    global_accuracy = np.array([])
    CYAN = '\033[96m'
    RC = '\033[0m'

    for task in tasks:

        subject_accuracy = np.array([])
        print(f"{CYAN}Task {task['name']}:{RC}")

        for subject in subjects:

            epochs = preprocessing(subject, task, mode)
            if mode == "train":
                accuracy = train(subject, task, epochs)
                subject_accuracy = np.append(subject_accuracy, accuracy)

        if mode == "train":
            mean_accuracy = np.mean(subject_accuracy)
            print(
                "Mean accuracy for task {}: {:.2f}%".format(
                    task['name'], mean_accuracy
                )
            )
            global_accuracy = np.append(global_accuracy, mean_accuracy)

    if mode == "train":
        mean_global_accuracy = np.mean(global_accuracy)
        print(f"Mean accuracy for all tasks: {mean_global_accuracy:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        print(exception)
    except KeyboardInterrupt:
        exit()
