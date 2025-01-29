"""Utility functions for similarity calculation."""

import numpy as np


def cosine_sim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the cosine similarity between x and y.

    Args:
        x: data 1. shape: (N, feats)
        y: data 2. shape: (N, feats)

    Return:
        cos similarity between x and y. shape: (N, )
    """
    assert (
        x.shape == y.shape
    ), f"x and y should have the same number of shape, but got {x.shape} and {y.shape}"
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-10)
    return np.sum(x * y, axis=1)


def weighted_corr_sim(
    x: np.ndarray, y: np.ndarray, corr: np.ndarray, dim: int
) -> np.ndarray:
    """Compute the weighted correlation similarity.

    Args:
        x: data 1. shape: (N, feats)
        y: data 2. shape: (N, feats)
        corr: correlation matrix. shape: (feats, )
        dim: number of dimensions to select

    Return:
        similarity matrix between x and y. shape: (N, )
    """
    assert (
        x.shape == y.shape
    ), f"x and y should have the same number of shape, but got {x.shape} and {y.shape}"
    # select the first dim dimensions
    x, y, corr = x[:, :dim], y[:, :dim], corr[:dim]
    # normalize x and y with L2 norm
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    # compute the similarity scores
    sim = np.zeros(x.shape[0])
    for ii in range(x.shape[0]):
        sim[ii] = corr * x[ii] @ y[ii]
    return sim


def batch_weighted_corr_sim(
    x: np.ndarray, y: np.ndarray, corr: np.ndarray, dim: int
) -> np.ndarray:
    """Compute the weighted correlation similarity for a batch of data, enabling batch processing.

    Args:
        x: data 1. shape: (N, feats)
        y: data 2. shape: (N, feats)
        corr: correlation matrix. shape: (feats, )
        dim: number of dimensions to select

    Return:
        similarity matrix between x and y. shape: (N, )
    """
    assert (
        x.shape == y.shape
    ), f"x and y should have the same number of shape, but got {x.shape} and {y.shape}"
    # select the first dim dimensions
    x, y, corr = x[:, :dim], y[:, :dim], corr[:dim]
    # normalize x and y with L2 norm
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    x = np.sqrt(corr) * x
    y = np.sqrt(corr) * y
    return np.sum(x * y, axis=1)
