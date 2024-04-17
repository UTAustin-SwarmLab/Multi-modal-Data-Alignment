"""Utility functions for similarity calculation."""

import numpy as np
import torch
from scipy import stats
from transformers import (
    AutoModel,
)


def clip_like_sim(
    model: AutoModel, text_features: np.ndarray, other_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the similarity score between text and other features using CLIP-like method.

    Args:
        model: CLIP-like model
        text_features: text features. shape: (N, D)
        other_features: other features. shape: (M, D)

    Returns:
        logits_per_text: similarity score between text and other features. shape: (N, M)
        logits_per_audio: similarity score between other and text features. shape: (M, N)
    """
    logit_scale_text = model.logit_scale_t.exp()
    logit_scale_audio = model.logit_scale_a.exp()
    logits_per_text = torch.matmul(text_features, other_features.t()) * logit_scale_text
    logits_per_audio = torch.matmul(other_features, text_features.t()) * logit_scale_audio
    return logits_per_text, logits_per_audio


def cosine_sim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the cosine similarity between x and y.

    Args:
        x: data 1. shape: (N, feats)
        y: data 2. shape: (N, feats)

    Return:
        cos similarity between x and y. shape: (N, )
    """
    assert x.shape == y.shape, f"x and y should have the same number of shape, but got {x.shape} and {y.shape}"
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.sum(x * y, axis=1)


def weighted_corr_sim(x: np.ndarray, y: np.ndarray, corr: np.ndarray, dim: int = 150) -> np.ndarray:
    """Compute the weighted correlation similarity.

    Args:
        x: data 1. shape: (N, feats)
        y: data 2. shape: (N, feats)
        corr: correlation matrix. shape: (feats, )
        dim: number of dimensions to select

    Return:
        similarity matrix between x and y. shape: (N, )
    """
    assert x.shape == y.shape, f"x and y should have the same number of shape, but got {x.shape} and {y.shape}"
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


def cal_roc_components(
    sim_align: np.ndarray, sim_unalign: np.ndarray, threshold: float
) -> tuple[float, float, float, float]:
    """Calculate the precision and recall.

    Args:
        sim_align: similarity score of aligned case. shape: (N, )
        sim_unalign: similarity score of unaligned case. shape: (N, )
        threshold: threshold
    Return:
        tp, fp, fn, tn
    """
    # positive is aligned, negative is unaligned
    tp = np.sum(sim_align > threshold)
    fp = np.sum(sim_unalign > threshold)
    fn = np.sum(sim_align <= threshold)
    tn = np.sum(sim_unalign <= threshold)
    assert (
        sim_align.shape[0] + sim_unalign.shape[0] == tp + fp + fn + tn
    ), f"tp + fp + fn + tn should be the number of samples, but got {tp + fp + fn + tn} \
        and {sim_align.shape[0] + sim_unalign.shape[0]}"
    return tp, fp, fn, tn


def roc_align_unalign_points(
    sim_align: np.ndarray,
    sim_unalign: np.ndarray,
    threshold_range: tuple[float, float, float] = (-1, 1, 40),
) -> list[tuple[float, float]]:
    """Calculate the roc points.

    Args:
        sim_align: similarity score of aligned case. shape: (N, )
        sim_unalign: similarity score of unaligned case. shape: (N, )
        threshold_range: threshold range. (start, end, points)

    Return:
        list of roc points
    """
    roc = []
    threshold_list = list(np.linspace(threshold_range[0], threshold_range[1], threshold_range[2]).reshape(-1))
    threshold_list += [-1, 1]
    threshold_list.sort()
    for threshold in threshold_list:
        tp, fp, fn, tn = cal_roc_components(sim_align, sim_unalign, threshold)
        tpr = tp / (tp + fn)  # y axis
        fpr = fp / (fp + tn)  # x axis
        roc.append((fpr, tpr))
    return roc


def cal_auc(roc_points: list[tuple[float, float]]) -> float:
    """Calculate the auc.

    Args:
        roc_points: list of roc points
    Return:
        auc (Area Under Curve)
    """
    roc_points = sorted(roc_points, key=lambda x: x[0])
    auc = 0
    for ii in range(1, len(roc_points)):
        auc += (roc_points[ii][0] - roc_points[ii - 1][0]) * (roc_points[ii][1] + roc_points[ii - 1][1]) / 2
    return auc


def spearman_rank_coefficient(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculate the Spearman rank correlation coefficient.

    Args:
        x: score of data 1. shape: (N,)
        y: score of data 2. shape: (N,)

    Return:
        Spearman rank correlation coefficient
    """
    assert x.shape == y.shape, f"x and y should have the same number of shape, but got {x.shape} and {y.shape}"
    n = x.shape[0]
    rank_x = np.argsort(x)
    rank_y = np.argsort(y)
    print(rank_x, rank_y)
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


if __name__ == "__main__":
    test_x = np.random.rand(1000, 150) * 2 - 1
    test_y = np.random.rand(1000, 150) * 2 - 1
    cossim = cosine_sim(test_x, test_y)
    print(cossim.shape, cossim.min(), cossim.max())
