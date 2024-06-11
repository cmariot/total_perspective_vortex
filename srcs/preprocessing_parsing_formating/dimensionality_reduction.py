import numpy as np
from sklearn.decomposition import PCA


def normalize_min_max(data: np.ndarray):
    for i, col in enumerate(data):
        data[i] = (col - col.min()) / (col.max() - col.min())
    return data


def dimensionality_reduction(x: np.ndarray):

    # Get y

    # Normalize
    normalize_min_max(x)

    # PCA
    # pca = PCA()

    # Split train/validation/test datasets


