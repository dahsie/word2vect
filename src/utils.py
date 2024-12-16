import numpy as np


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two samples.

    Parameters:
        x1 (np.ndarray): First sample.
        x2 (np.ndarray): Second sample.

    Returns:
        float: Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2)**2, axis = 0))

def cosinus_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculates Cosine similarity between two samples.

    Parameters:
        x1 (np.ndarray): First sample.
        x2 (np.ndarray): Second sample.

    Returns:
        float: Cosine similarity between x1 and x2.
    """

    return x1.dot(x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))