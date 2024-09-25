"""Calibration for the similarity matrix."""

import warnings
from bisect import bisect_left


def get_non_conformity_scores(
    sim_mat: dict, idx_modal1: int, idx_modal2: int
) -> tuple[list[float], list[float]]:
    """Get the (non)conformity scores.

    Args:
        sim_mat: the similarity matrix. dict: (i, j) -> (similarity matrix, ground truth)
        idx_modal1: index of the first modality
        idx_modal2: index of the second modality

    Returns:
        nc_scores: the nonconformity scores (modal_a-modal_b and modal_b-modal_a are the same)
        c_scores: the conformity scores (modal_a-modal_b and modal_b-modal_a are the same)
    """
    # raise warning if i > j
    if idx_modal1 > idx_modal2:
        msg = f"idx_modal1 ({idx_modal1}) > idx_modal2 ({idx_modal2})"
        warnings.warn(msg, stacklevel=2)

    nc_scores, c_scores = [], []
    for mat, label in sim_mat.values():
        if label == 1:  # positive pair
            c_scores.append(mat[idx_modal1][idx_modal2])
            if idx_modal1 != idx_modal2:  # symmetry of entries
                c_scores.append(mat[idx_modal2][idx_modal1])
        elif label == 0:  # negative pair
            nc_scores.append(mat[idx_modal1][idx_modal2])
            if idx_modal1 != idx_modal2:  # symmetry of entries
                nc_scores.append(mat[idx_modal2][idx_modal1])
        else:  # invalid label
            msg = f"Invalid label: {label}"
            raise ValueError(msg)
    # sort the scores in ascending order
    nc_scores.sort()
    c_scores.sort()
    assert nc_scores[0] < nc_scores[-1]
    assert c_scores[0] < c_scores[-1]
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
    # we can use binary search to find the index of score in nc_scores
    idx = bisect_left(nc_scores, score)
    # the calibrated score is the index divided by the total number of scores
    return idx / len(nc_scores)
