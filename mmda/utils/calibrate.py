"""Calibration for the similarity matrix."""

from bisect import bisect_left

import numpy as np


def get_non_conformity_scores(
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
    return nc_scores, c_scores


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
