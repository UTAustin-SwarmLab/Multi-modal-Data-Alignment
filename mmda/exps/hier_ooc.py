"""Use a hierarchical decision making process to predict out-of-context image and captions.

The setting is, given a set of corresponding images and captions, predict the out-of-context of a new caption.
"""

import numpy as np
from omegaconf import DictConfig

from mmda.utils.data_utils import (
    load_clip_like_data,
)
from mmda.utils.ooc_dataset_class import load_hier_dataset
from mmda.utils.sim_utils import (
    cosine_sim,
)


def cca_hier_ooc(
    cfg: DictConfig,
) -> list[tuple[float, float]]:
    """Hierarchical decision making process to predict out-of-context image and captions using the proposed CCA method.

    Args:
        cfg: configuration file
    Returns:
        ROC_points: ROC points
    """


def clip_like_hier_ooc(
    cfg: DictConfig,
) -> list[tuple[float, float]]:
    """Hierarchical decision making process to predict out-of-context image and captions using the CLIP-like method.

    Args:
        cfg: configuration file
    Returns:
        ROC_points: ROC points
    """
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, data1, data2 = load_clip_like_data(cfg)

    hier_ds = load_hier_dataset(cfg)
    hier_ds.preprocess_retrieval_data(data1, data2)
    hier_ds.set_similarity_metrics(cosine_sim, cosine_sim)
    return hier_ds.detect_ooc()  # tp, fp, fn, tn


def asif_hier_ooc(
    cfg: DictConfig,
) -> list[tuple[float, float]]:
    """Hierarchical decision making process to predict out-of-context image and captions using the ASIF method.

    Args:
        cfg: configuration file
    Returns:
        ROC_points: ROC points
    """
