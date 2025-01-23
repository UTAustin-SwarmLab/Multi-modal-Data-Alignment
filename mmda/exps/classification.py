"""This module contains the functions to detect mislabeled data using the proposed method and baselines."""

import numpy as np
from omegaconf import DictConfig

from mmda.utils.cca_class import NormalizedCCA, ReNormalizedCCA
from mmda.utils.classification_dataset_class import load_classification_dataset
from mmda.utils.sim_utils import cosine_sim, weighted_corr_sim


def cca_classification(
    cfg: DictConfig, train_test_ratio: float, shuffle_ratio: float = 0.0
) -> float:
    """Retrieve data using the proposed CCA method.

    Args:
        cfg: configuration file
        train_test_ratio: ratio of training data
        shuffle_ratio: ratio of shuffling data
    Returns:
        data_size2accuracy: {data_size: accuracy}
    """
    cfg_dataset = cfg[cfg.dataset]
    print(f"CCA {cfg_dataset.sim_dim}")
    ds = load_classification_dataset(cfg)
    ds.load_data(train_test_ratio, clip_bool=False, shuffle_ratio=shuffle_ratio)
    cca = ReNormalizedCCA() if True else NormalizedCCA()
    ds.train_img, ds.train_text, corr = cca.fit_transform_train_data(
        cfg_dataset, ds.train_img, ds.train_text
    )
    ds.test_img, ds.test_text = cca.transform_data(ds.test_img, ds.test_text)

    ds.get_labels_emb()
    dummy_img_emb = np.zeros_like(ds.img_emb)[: ds.labels_emb.shape[0], :]
    dummy_img_emb, ds.labels_emb = cca.transform_data(dummy_img_emb, ds.labels_emb)

    def sim_fn(x: np.array, y: np.array, corr: np.array = corr) -> np.array:
        return weighted_corr_sim(x, y, corr=corr, dim=cfg_dataset.sim_dim)

    return ds.classification(sim_fn=sim_fn)


def clip_like_classification(cfg: DictConfig, train_test_ratio: float) -> float:
    """Retrieve data using the CLIP-like method.

    Args:
        cfg: configuration file
        train_test_ratio: ratio of training data
    Returns:
        data_size2accuracy: {data_size: accuracy}
    """
    print("CLIP-like")
    ds = load_classification_dataset(cfg)
    ds.load_data(train_test_ratio, clip_bool=True)
    ds.get_labels_emb()
    return ds.classification(sim_fn=cosine_sim)


def asif_classification(
    cfg: DictConfig, train_test_ratio: float, shuffle_ratio: float = 0.0
) -> float:
    """Retrieve data using the CLIP-like method.

    Args:
        cfg: configuration file
        train_test_ratio: ratio of training data
        shuffle_ratio: ratio of shuffling data
    Returns:
        data_size2accuracy: {data_size: accuracy}
    """
    print("ASIF")
    ds = load_classification_dataset(cfg)
    ds.load_data(train_test_ratio, clip_bool=False, shuffle_ratio=shuffle_ratio)
    ds.get_labels_emb()
    return ds.classification(sim_fn="asif")
