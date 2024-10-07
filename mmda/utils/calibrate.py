"""Calibration for the similarity matrix."""

from bisect import bisect_left

import numpy as np
from tqdm import tqdm


def con_mat_calibrate(sim_mat_dict: dict, pred_band: dict) -> dict:
    """Calculate the conformal matrix from similarity matirx.

    Args:
        sim_mat_dict: the similarity matrix. dict: (i, j) -> (similarity matrix, ground truth)
        pred_band: the prediction band. dict: (i, j) -> callable

    Returns:
        The conformal matrix. dict: (i, j) -> (conformal matrix, ground truth)
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
                probs[i, j] = pred_band[(min(i, j), max(i, j))](sim_mat[i, j])
        con_mat[(idx_q, idx_r)] = (probs, gt_label)
    return con_mat


def get_non_conformity_scores_common(
    q2scores: dict, top_k: int = 5
) -> tuple[list[float], list[float]]:
    """Get the nonconformity scores for the quenry index dict."""
    for q in q2scores:
        q2scores[q] = np.array(q2scores[q])
    for q in q2scores:
        q2scores[q] = q2scores[q][np.argsort(-q2scores[q][:, 0])]
        q2scores[q] = (
            q2scores[q][top_k - 1][0],  # the score of the top k-th pair
            int(
                sum(q2scores[q][:top_k, 1]) >= 1.0
            ),  # if there is at least one positive pair in the top k
        )
    nc_scores = []
    c_scores = []
    for q in q2scores:
        if q2scores[q][1] == 1.0:
            c_scores.append(q2scores[q][0])
        else:
            nc_scores.append(q2scores[q][0])
    # sort the scores
    nc_scores.sort()
    c_scores.sort()
    return nc_scores, c_scores, q2scores


def get_non_conformity_scores_1st_stage(
    sim_mat: dict, idx_modal1: int, idx_modal2: int, top_k: int = 5
) -> tuple[list[float], list[float]]:
    """Get the nonconformity scores for the given modalities.

    Args:
        sim_mat: the similarity matrix. dict: (i, j) -> (similarity matrix, ground truth)
        idx_modal1: index of the first modality
        idx_modal2: index of the second modality
        top_k: the number of top scores to consider for recall

    Returns:
        nc_scores: the nonconformity scores
        c_scores: the conformity scores
        q2scores: the query index dict i -> (top k-th sim score, label)
    """
    q2scores = {}  # dict: query idx -> (top k-th sim score, label)
    for q, r in sim_mat:
        # skip the same pair of query and reference data
        if q == r:
            continue
        if q not in q2scores:
            q2scores[q] = []
        q2scores[q].append((sim_mat[q, r][0][idx_modal1][idx_modal2], sim_mat[q, r][1]))
        if r not in q2scores:
            q2scores[r] = []
        q2scores[r].append((sim_mat[q, r][0][idx_modal1][idx_modal2], sim_mat[q, r][1]))
    return get_non_conformity_scores_common(q2scores, top_k)


def get_non_conformity_scores_2nd_stage(
    con_mat: dict, mapping_fn: callable, top_k: int = 5
) -> tuple[list[float], list[float]]:
    """Get the nonconformity scores for the given conformal matrices.

    Args:
        con_mat: the conformal matrix. dict: (i, j) -> (conformal matrix, ground truth)
        mapping_fn: the mapping function to map the conformal matrix to a scalar, e.g., max, mean
        top_k: the number of top scores to consider for recall

    Returns:
        nc_scores: the nonconformity scores
        c_scores: the conformity scores
        q2scores: the query index dict i -> (top k-th sim score, label)
    """
    q2scores = {}  # dict: query idx -> (top k-th sim score, label)
    for q, r in con_mat:
        # skip the same pair of query and reference data
        if q == r:
            continue
        if q not in q2scores:
            q2scores[q] = []
        q2scores[q].append((mapping_fn(con_mat[q, r][0]), con_mat[q, r][1]))
        if r not in q2scores:
            q2scores[r] = []
        q2scores[r].append((mapping_fn(con_mat[q, r][0]), con_mat[q, r][1]))
    return get_non_conformity_scores_common(q2scores, top_k)


def calibrate(score: float, nc_scores: list[float]) -> float:
    """Calibrate the score using the nonconformity scores.

    Args:
        score: the score to be calibrated
        nc_scores: the nonconformity scores

    Returns:
        The calibrated score.
    """
    # see score is bigger than how many scores in nc_scores
    # since nc_scores is sorted in ascending order,
    # the calibrated score is the index divided by the total number of scores
    # we can use binary search to find the index of score in nc_scores
    assert nc_scores[0] <= nc_scores[-1], "nc_scores should be in ascending order"
    return bisect_left(nc_scores, score) / len(nc_scores)
