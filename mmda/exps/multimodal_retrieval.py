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


def cca_retrieval(cfg: DictConfig) -> tuple[dict[float:float], dict[float:float]]:
    """Retrieve data using the proposed CCA method.

    Args:
        cfg: configuration file
    Returns:
        recalls: {1: recall@1, 5:recall@5} if img2text else {1:recall@1}
        precisions: {1: precision@1, 5:precision@5} if img2text else {1:precision@1}
    """
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)

    retreival_dataset = load_retrieval_dataset(cfg)
    retreival_dataset.preprocess_retrieval_data(data1, data2)

    cca, retreival_dataset.traindata1, retreival_dataset.traindata2, corr = (
        cca_fit_train_data(
            cfg_dataset, retreival_dataset.traindata1, retreival_dataset.traindata2
        )
    )
    retreival_dataset.testdata1, retreival_dataset.testdata2 = cca.transform(
        (retreival_dataset.testdata1, retreival_dataset.testdata2)
    )
    recalls, precisions = {}, {}
    for cca_proj_dim in cfg_dataset.cca_proj_dims:

        def sim_fn(x: np.array, y: np.array) -> np.array:
            return weighted_corr_sim(x, y, corr, dim=cca_proj_dim)  # noqa: B023

        recall, precision = retreival_dataset.recall_presicion_at_k(sim_fn=sim_fn)
        recalls[cca_proj_dim], precisions[cca_proj_dim] = recall, precision
    return recalls, precisions


def clip_like_retrieval(cfg: DictConfig) -> tuple[dict[float:float], dict[float:float]]:
    """Retrieve data using the CLIP-like method.

    Args:
        cfg: configuration file
    Returns:
        recalls: {1: recall@1, 5:recall@5} if img2text else {1:recall@1}
        precisions: {1: precision@1, 5:precision@5} if img2text else {1:precision@1}
    """
    cfg_dataset, data1, data2 = load_clip_like_data(cfg)
    retreival_dataset = load_retrieval_dataset(cfg)
    retreival_dataset.preprocess_retrieval_data(data1, data2)
    recalls, precisions = retreival_dataset.recall_presicion_at_k(sim_fn=cosine_sim)
    return recalls, precisions
