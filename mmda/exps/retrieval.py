"""This module contains the functions to detect mislabeled data using the proposed method and baselines."""

import numpy as np
from omegaconf import DictConfig

from mmda.utils.cca_utils import cca_fit_train_data
from mmda.utils.data_utils import (
    load_clip_like_data,
    load_two_encoder_data,
)
from mmda.utils.retrieval_dataset_class import load_retrieval_dataset
from mmda.utils.sim_utils import (
    cosine_sim,
    weighted_corr_sim,
)


def cca_retrieval(
    cfg: DictConfig,
) -> tuple[dict[float:float], dict[float : dict[float:float]]]:
    """Retrieve data using the proposed CCA method.

    Args:
        cfg: configuration file
    Returns:
        map: mAP
        precisions: {1: precision@1, 5:precision@5}
    """
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)

    retrieval_ds = load_retrieval_dataset(cfg)
    retrieval_ds.preprocess_retrieval_data(data1, data2)

    cca, retrieval_ds.traindata1, retrieval_ds.traindata2, corr = cca_fit_train_data(
        cfg_dataset, retrieval_ds.traindata1, retrieval_ds.traindata2
    )
    retrieval_ds.testdata1, retrieval_ds.testdata2 = cca.transform(
        (retrieval_ds.testdata1, retrieval_ds.testdata2)
    )
    maps, precisions = {}, {}
    for cca_proj_dim in cfg_dataset.cca_proj_dims:

        def sim_fn(x: np.array, y: np.array) -> np.array:
            return weighted_corr_sim(x, y, corr, dim=cca_proj_dim)  # noqa: B023

        map_, precision = retrieval_ds.top_k_presicion(sim_fn=sim_fn)
        maps[cca_proj_dim], precisions[cca_proj_dim] = map_, precision
    return maps, precisions


def clip_like_retrieval(cfg: DictConfig) -> tuple[dict[float:float], dict[float:float]]:
    """Retrieve data using the CLIP-like method.

    Args:
        cfg: configuration file
    Returns:
        maps: {1: recall@1, 5:recall@5} if img2text else {1:recall@1}
        precisions: {1: precision@1, 5:precision@5} if img2text else {1:precision@1}
    """
    cfg_dataset, data1, data2 = load_clip_like_data(cfg)
    retrieval_ds = load_retrieval_dataset(cfg)
    retrieval_ds.preprocess_retrieval_data(data1, data2)
    return retrieval_ds.top_k_presicion(sim_fn=cosine_sim)
