import numpy as np


def weighted_corr_sim(X: np.ndarray, Y: np.ndarray, corr: np.ndarray, dim: int=150) -> np.ndarray:
    """
    Compute the weighted correlation similarity
    :param X: data 1. shape: (N, feats)
    :param Y: data 2. shape: (N, feats)
    :param corr: correlation matrix. shape: (feats, )
    :return: similarity matrix between X and Y. shape: (N, )
    """

    assert X.shape == Y.shape, f"X and Y should have the same number of shape, but got {X.shape} and {Y.shape}"
    # select the first dim dimensions
    X, Y, corr = X[:, :dim], Y[:, :dim], corr[:dim]
    # normalize X and Y with L2 norm
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    # compute the similarity scores
    sim = np.zeros(X.shape[0])
    # for ii in range(X.shape[0]):
    #     sim[ii] = corr * X[ii] @ Y[ii]
    sim = corr * np.diag(X @ Y.T)
    return sim
    