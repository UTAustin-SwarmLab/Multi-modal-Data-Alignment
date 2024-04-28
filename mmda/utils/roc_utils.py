"""This module contains the functions to calculate the ROC curve and AUC."""

import numpy as np


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
    roc = [(0.0, 0.0), (1.0, 1.0)]
    threshold_list = list(
        np.linspace(threshold_range[0], threshold_range[1], threshold_range[2]).reshape(
            -1
        )
    )
    for threshold in threshold_list:
        tp, fp, fn, tn = cal_roc_components(sim_align, sim_unalign, threshold)
        tpr = tp / (tp + fn)  # y axis
        fpr = fp / (fp + tn)  # x axis
        roc.append((fpr, tpr))
    # keep only the unique points
    roc = list(set(roc))
    return sorted(roc, key=lambda x: x[0] + x[1])


def cal_auc(roc_points: list[tuple[float, float]]) -> float:
    """Calculate the auc.

    Args:
        roc_points: list of roc points
    Return:
        auc (Area Under Curve)
    """
    auc = 0
    for ii in range(1, len(roc_points)):
        auc += (
            (roc_points[ii][0] - roc_points[ii - 1][0])
            * (roc_points[ii][1] + roc_points[ii - 1][1])
            / 2
        )
    return auc
