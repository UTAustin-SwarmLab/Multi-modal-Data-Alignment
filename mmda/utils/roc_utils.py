"""This module contains the functions to calculate the ROC curve and AUC."""

import numpy as np
from scipy.spatial import ConvexHull


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
    threshold_list = list(
        np.linspace(threshold_range[0], threshold_range[1], threshold_range[2]).reshape(
            -1
        )
    )
    tps = []
    for threshold in threshold_list:
        tp, fp, fn, tn = cal_roc_components(sim_align, sim_unalign, threshold)
        tps.append((tp, fp, fn, tn))
    return tp_fp_fn_tn_to_roc(tps)


def tp_fp_fn_tn_to_roc(
    tps: list[tuple[float, float, float, float]],
) -> list[tuple[float, float]]:
    """Calculate the roc points from true positives etc.

    Args:
        tps: tuples of (true positives, false positives false negatives, and true negatives)

    Return:
        list of roc points
    """
    roc_points = [(0.0, 0.0), (1.0, 1.0)]
    for tp, fp, fn, tn in tps:
        tpr = tp / (tp + fn)  # y axis
        fpr = fp / (fp + tn)  # x axis
        roc_points.append((fpr, tpr))
    # keep only the unique points
    roc_points = list(set(roc_points))
    return sorted(roc_points, key=lambda x: x[0])


def select_maximum_auc(
    dict_tps: dict[tuple[float, float], tuple[int, int, int, int]]
) -> list[float, float]:
    """Select the threshold that gives the maximum AUC.

    Args:
        dict_tps: dictionary of threshold and tp, fp, fn, tn
    Returns:
        roc_points: ROC points
    """
    max_auc = 0
    max_roc_points = None
    dict_texts_threshold, dict_text_image_threshold = zip(
        *dict_tps.keys(), strict=False
    )
    # fix texts_threshold and change text_image_threshold
    for texts_threshold in dict_texts_threshold:
        tps_list = []
        for text_image_threshold in dict_text_image_threshold:
            tps = dict_tps[(texts_threshold, text_image_threshold)]
            tps_list.append(tps)
        roc_points = tp_fp_fn_tn_to_roc(tps_list)
        auc = cal_auc(roc_points)
        if auc > max_auc:
            max_auc = auc
            max_roc_points = roc_points
    # fix text_image_threshold and change texts_threshold
    for text_image_threshold in dict_text_image_threshold:
        tps_list = []
        for texts_threshold in dict_texts_threshold:
            tps = dict_tps[(texts_threshold, text_image_threshold)]
            tps_list.append(tps)
        roc_points = tp_fp_fn_tn_to_roc(tps_list)
        auc = cal_auc(roc_points)
        if auc > max_auc:
            max_auc = auc
            max_roc_points = roc_points
    return max_roc_points


def convex_hull_roc_points(
    dict_tps: dict[tuple[float, float], tuple[int, int, int, int]]
) -> list[float, float]:
    """Get the convex hull of the ROC points.

    Args:
        dict_tps: dictionary of threshold and tp, fp, fn, tn
    Returns:
        roc_points: convex hull of the ROC points
    """
    list_tps = dict_tps.values()
    points = tp_fp_fn_tn_to_roc(list_tps)
    hull = ConvexHull(points)
    vertices_idx = hull.vertices
    roc_points = [points[i] for i in vertices_idx]
    return sorted(roc_points, key=lambda x: x[0])


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
