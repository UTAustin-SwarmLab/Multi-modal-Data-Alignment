"""Utility functions for Spearman rank correlation coefficient."""

import numpy as np
from scipy import stats


def spearman_rank_coefficient(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculate the Spearman rank correlation coefficient.

    Args:
        x: score of data 1. shape: (N,)
        y: score of data 2. shape: (N,)

    Return:
        Spearman rank correlation coefficient
        rank of x
        rank of y
    """
    assert (
        x.shape == y.shape
    ), f"x and y should have the same number of shape, but got {x.shape} and {y.shape}"
    n = x.shape[0]
    rank_x = np.argsort(x)
    rank_y = np.argsort(y)
    d = np.sum((rank_x - rank_y) ** 2)
    return 1 - 6 * d / (n * (n**2 - 1)), rank_x, rank_y


def spearman_to_p_value(r: float, n: int) -> float:
    """Calculate the p-value from Spearman rank correlation coefficient.

    Note that the calculations assume that the null hypothesis is true, i.e., there is no correlation.
    If the p-value is less than the chosen significance level (often 0.05), we would reject the null
    hypothesis and conclude that there is a significant correlation.

    Args:
        r: Spearman rank correlation coefficient
        n: number of samples

    Return:
        p-value
    """
    t = r * np.sqrt((n - 2) / (1 - r**2))
    return stats.t.sf(np.abs(t), n - 2) * 2
