from typing import List, Tuple

import numpy as np
import torch
from transformers import (
    AutoModel,
)


def CLIP_like_sim(model: AutoModel, text_features: np.ndarray, other_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate the similarity score between text and other features using CLIP-like method
    Args:
        model: CLIP-like model
        text_features: text features. shape: (N, D)
        other_features: other features. shape: (M, D)
    Returns:
        logits_per_text: similarity score between text and other features. shape: (N, M)
        logits_per_audio: similarity score between other and text features. shape: (M, N)
    '''
    logit_scale_text = model.logit_scale_t.exp()
    logit_scale_audio = model.logit_scale_a.exp()
    logits_per_text = torch.matmul(text_features, other_features.t()) * logit_scale_text
    logits_per_audio = torch.matmul(other_features, text_features.t()) * logit_scale_audio

    return logits_per_text, logits_per_audio

def cosine_sim(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between X and Y
    Args:
        X: data 1. shape: (N, feats)
        Y: data 2. shape: (N, feats)
    Return: 
        cos similarity between X and Y. shape: (N, )
    """
    assert X.shape == Y.shape, f"X and Y should have the same number of shape, but got {X.shape} and {Y.shape}"
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return np.sum(X * Y, axis=1)


def weighted_corr_sim(X: np.ndarray, Y: np.ndarray, corr: np.ndarray, dim: int=150) -> np.ndarray:
    """
    Compute the weighted correlation similarity
    Args:
        X: data 1. shape: (N, feats)
        Y: data 2. shape: (N, feats)
        corr: correlation matrix. shape: (feats, )
    Return: 
        similarity matrix between X and Y. shape: (N, )
    """

    assert X.shape == Y.shape, f"X and Y should have the same number of shape, but got {X.shape} and {Y.shape}"
    # select the first dim dimensions
    X, Y, corr = X[:, :dim], Y[:, :dim], corr[:dim]
    # normalize X and Y with L2 norm
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    # compute the similarity scores
    sim = np.zeros(X.shape[0])
    for ii in range(X.shape[0]):
        sim[ii] = corr * X[ii] @ Y[ii]
    return sim
    
def cal_ROC_components(sim_align: np.ndarray, sim_unalign: np.ndarray, threshold: float) -> Tuple[float, float, float, float]:
    """
    Calculate the precision and recall
    Args:
        sim_align: similarity score of aligned case. shape: (N, )
        sim_unalign: similarity score of unaligned case. shape: (N, )
        threshold: threshold 
    Return: 
        TP, FP, FN, TN
    """
    # positive = aligned, negative = unaligned
    TP = np.sum(sim_align > threshold)
    FP = np.sum(sim_unalign > threshold)
    FN = np.sum(sim_align <= threshold)
    TN = np.sum(sim_unalign <= threshold)
    assert TP + FP + FN + TN == sim_align.shape[0] + sim_unalign.shape[0], \
        f"TP + FP + FN + TN should be the number of samples, but got {TP + FP + FN + TN} and {sim_align.shape[0] + sim_unalign.shape[0]}"
    return TP, FP, FN, TN

def ROC_points(sim_align: np.ndarray, sim_unalign: np.ndarray, threshold_list: List[float]) -> List[Tuple[float, float]]:
    """
    Calculate the ROC points
    Args:
        sim_align: similarity score of aligned case. shape: (N, )
        sim_unalign: similarity score of unaligned case. shape: (N, )
        threshold_list: list of thresholds
    Return: 
        list of ROC points
    """
    ROC = []
    for threshold in threshold_list:
        TP, FP, FN, TN = cal_ROC_components(sim_align, sim_unalign, threshold)
        TPR = TP / (TP + FN)  # y axis
        FPR = FP / (FP + TN)  # x axis
        ROC.append((FPR, TPR))
    return ROC

def cal_AUC(ROC_points: List[Tuple[float, float]]) -> float:
    """
    Calculate the AUC
    Args:
        ROC_points: list of ROC points
    Return:
        AUC (Area Under Curve)
    """
    ROC_points = sorted(ROC_points, key=lambda x: x[0])
    AUC = 0
    for ii in range(1, len(ROC_points)):
        AUC += (ROC_points[ii][0] - ROC_points[ii-1][0]) * (ROC_points[ii][1] + ROC_points[ii-1][1]) / 2
    return AUC

if __name__ == '__main__':
    test_X = np.random.rand(1000, 150) * 2 - 1
    test_Y = np.random.rand(1000, 150) * 2 - 1
    cossim = cosine_sim(test_X, test_Y)
    print(cossim.shape, cossim.min(), cossim.max())

    # sim_align = 2 * np.random.rand(200) - 1
    # sim_unalign = 2 * np.random.rand(100) - 1 
    # threshold_list = np.linspace(-1, 1, 20)
    # points = ROC_points(sim_align, sim_unalign, threshold_list)
    # print(f"AUC: {cal_AUC(points)}")
