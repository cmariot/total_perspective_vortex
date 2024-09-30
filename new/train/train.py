try:

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from mne.decoding import CSP
    from sklearn.model_selection import ShuffleSplit, cross_val_score

except ImportError:
    raise ImportError(
        "Please the required dependencies" +
        " by running: pip install -r requirements.txt"
    )


def train(subject, task, epochs):

    # Get labels
    labels = epochs.events[:, -1] - 2

    # CLASSIFICATION

    cv = ShuffleSplit(n_splits=10, random_state=42, test_size=0.2)
    # epochs_data = 1e6 * epochs.get_data()
    epochs_data = epochs.get_data()

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)

    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    accuracy = np.mean(scores) * 100
    print(
        f"{CYAN}Subject {subject:03d} - " +
        f"Accuracy: {accuracy:.2f}%{ENDC}"
    )

    return accuracy
