"""Utility functions for similarity calculation."""

import numpy as np
import torch
from transformers import AutoModel


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
    logits_per_audio = (
        torch.matmul(other_features, text_features.t()) * logit_scale_audio
    )
    return logits_per_text, logits_per_audio


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
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
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
