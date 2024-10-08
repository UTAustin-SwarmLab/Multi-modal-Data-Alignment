from pathlib import Path

import joblib
import numpy as np

import hydra
from mmda.utils.calibrate import (
    calibrate,
    con_mat_calibrate,
    get_calibration_scores_1st_stage,
    get_calibration_scores_2nd_stage,
)


def load_sim_mat_cali(
    config_path: str = "../config", config_name: str = "main"
) -> dict:
    """Load the similarity matrix for calibration."""
    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name=config_name)
        cfg_dataset = cfg["KITTI"]
        return joblib.load(
            Path(
                cfg_dataset.paths.save_path,
                f"sim_mat_cali_{cfg_dataset.retrieval_dim}_{cfg_dataset.mask_ratio}.pkl",
            )
        )


def overlap_ratio(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Computes the overlap ratio between two histograms."""
    minima = np.minimum(hist1, hist2)
    intersection = np.sum(minima)
    union = np.sum(hist1) + np.sum(hist2) - intersection
    return intersection / union


def get_scores_and_gt(
    data: dict, idx_modal1: int, idx_modal2: int
) -> tuple[list[float], list[float]]:
    """Get the scores and ground truth for the given modalities."""
    scores = []
    gt = []
    for mat, label in data.values():
        scores.append(mat[idx_modal1][idx_modal2])
        gt.append(label)
    return scores, gt


IMAGE = 0
LIDAR = 1
TEXT = 2

bin_edges = np.array(range(51)) / 25 - 1
modalities = ["Image", "Lidar", "Text"]

data = load_sim_mat_cali()


scores3x3 = {}
for i in range(3):
    for j in range(3):
        scores, _ = get_calibration_scores_1st_stage(data, i, j)
        scores3x3[(i, j)] = scores

con_mat = con_mat_calibrate(data, scores3x3)
scores, gts = get_calibration_scores_2nd_stage(con_mat, np.mean)
print(len(scores))
cali_scores = []
for score in scores:
    cali_scores.append(calibrate(score, scores))
print(cali_scores)
