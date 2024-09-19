"""Calibration for the similarity matrix."""

import warnings


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

    return nc_scores, c_scores


def calibrate(score: float, nc_scores: list[float]) -> float:
    """Calibrate the score using the nonconformity scores.

    Args:
        score: the score to be calibrated
        nc_scores: the nonconformity scores

    Returns:
        The calibrated score.
    """
    count = 0
    for i in nc_scores:
        if i < score:
            count += 1
    return count / len(nc_scores)
