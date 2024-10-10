"""Calibration for the similarity matrix."""

from bisect import bisect_left

import numpy as np
from tqdm import tqdm


def calibration_score(
    x: np.ndarray, gt: np.ndarray, min_score: float, max_score: float
) -> np.ndarray:
    """Calculate the calibration score.

    Args:
        x: the similarity score. shape: (n)
        gt: the ground truth label. shape: (n)
        min_score: the minimum score. shape: (n)
        max_score: the maximum score. shape: (n)

    Returns:
        The calibration score. shape: (n)
    """
    return np.abs(gt - (x - min_score) / (max_score - min_score))


def con_mat_calibrate(sim_mat_dict: dict, scores: dict) -> dict:
    """Calculate the conformal matrix from similarity matirx.

    Args:
        sim_mat_dict: the similarity matrix. dict: (i, j) -> (similarity matrix, ground truth)
        scores: the nonconformity scores. dict: (i, j) -> list[float]

    Returns:
        con_mat: The conformal matrix. dict: (i, j) -> (conformal matrix, ground truth)
    """
    con_mat = {}
    for (idx_q, idx_r), (sim_mat, gt_label) in tqdm(
        sim_mat_dict.items(),
        desc="Calculating conformal probabilities",
        leave=True,
    ):
        probs = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                probs[i, j] = calibrate(sim_mat[i, j], scores[(i, j)])
        con_mat[(idx_q, idx_r)] = (probs, gt_label)
    return con_mat


def get_calibration_scores_common(
    sim_scores: list[float], gt_labels: list[int]
) -> tuple[list[float], list[int]]:
    """Get the nonconformity scores for the given modalities.

    Args:
        sim_scores: the similarity scores
        gt_labels: the ground truth labels

    Returns:
        cali_scores: the SORTED distribution of calibration scores
        gt_labels: the ground truth labels
    """
    sim_scores = np.array(sim_scores).reshape(-1)
    gt_labels = np.array(gt_labels).reshape(-1)
    # calculate the calibration scores
    cali_scores = calibration_score(
        np.array(sim_scores), np.array(gt_labels), min(sim_scores), max(sim_scores)
    )
    assert cali_scores.shape == sim_scores.shape
    # ascending order
    return sorted(cali_scores.reshape(-1).tolist()), gt_labels


def get_calibration_scores_1st_stage(
    sim_mat_dict: dict, idx_modal1: int, idx_modal2: int, skip_same: bool = False
) -> tuple[list[float], list[int]]:
    """Get the nonconformity scores for the given modalities.

    Args:
        sim_mat_dict: the similarity matrix. dict: (i, j) -> (similarity matrix, ground truth)
        idx_modal1: index of the first modality
        idx_modal2: index of the second modality
        skip_same: whether to skip the same pair of query and reference data (default: False)

    Returns:
        cali_scores: the SORTED distribution of calibration scores
        gt_labels: the ground truth labels
    """
    sim_scores = []
    gt_labels = []
    for (idx_q, idx_r), (sim_mat, gt_label) in sim_mat_dict.items():
        # skip the same pair of query and reference data
        if idx_q == idx_r and skip_same:
            continue
        sim_scores.append(sim_mat[idx_modal1, idx_modal2])
        gt_labels.append(gt_label)
    return get_calibration_scores_common(sim_scores, gt_labels)


def get_calibration_scores_2nd_stage(
    con_mat: dict,
    mapping_fn: callable,
    skip_same: bool = False,
) -> tuple[list[float], list[int]]:
    """Get the nonconformity scores for the given conformal matrices.

    Args:
        con_mat: the conformal matrix. dict: (i, j) -> (conformal matrix, ground truth)
        mapping_fn: the mapping function to map the conformal matrix to a scalar, e.g., max, mean
        skip_same: whether to skip the same pair of query and reference data (default: False)

    Returns:
        cali_scores: the SORTED distribution of calibration scores.
        gt_labels: the ground truth labels.
    """
    sim_scores = []
    gt_labels = []
    for idx_q, idx_r in con_mat:
        # skip the same pair of query and reference data
        if idx_q == idx_r and skip_same:
            continue
        con_mat_ = con_mat[(idx_q, idx_r)][0]
        # skip the empty conformal matrix
        if len(con_mat_[con_mat_ != -1].reshape(-1)) == 0:
            continue
        sim_scores.append(mapping_fn(con_mat_[con_mat_ != -1]))
        gt_labels.append(con_mat[(idx_q, idx_r)][1])
    return get_calibration_scores_common(sim_scores, gt_labels)


def calibrate(score: float, scores: list[float]) -> float:
    """Calibrate the score using the nonconformity scores.

    Args:
        score: the score to be calibrated
        scores: the nonconformity scores

    Returns:
        The calibrated score.
    """
    # see score is bigger than how many scores in nc_scores
    # since nc_scores is sorted in ascending order,
    # the calibrated score is the index divided by the total number of scores
    # we can use binary search to find the index of score in nc_scores
    assert scores[0] <= scores[-1], "nc_scores should be in ascending order"
    return bisect_left(scores, score) / len(scores)
