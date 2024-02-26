import numpy as np
from typing import List, Tuple

def origin_centered(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ''' This function returns the origin centered data matrix and the mean of each feature
    Args:
        X: data matrix (n_samples, n_features)
    Returns: 
        origin centered data matrix, mean of each feature
    '''
    return X - np.mean(X, axis=0), np.mean(X, axis=0)