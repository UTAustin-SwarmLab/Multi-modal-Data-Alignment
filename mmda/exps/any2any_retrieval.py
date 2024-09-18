"""This module contains the functions to detect mislabeled data using the proposed method and baselines."""

from omegaconf import DictConfig

from mmda.utils.retrieval_dataset_class import load_retrieval_dataset


def any2any_retrieval(cfg: DictConfig) -> tuple[list, list, list]:
    """Retrieve the multimodal datasets using the any2any conformal retrieval.

    Args:
        cfg: configuration file

    Returns:
        mAPs: mean average precision scores
        precisions: precision at 1, 5, 20
        recalls: recall at 1, 5, 20
    """
    cfg_dataset = cfg[cfg.dataset]
    ds = load_retrieval_dataset(cfg)
    ds.preprocess_retrieval_data()
    ds.train_crossmodal_similarity()
    if cfg_dataset.calibrate:
        ds.calibrate_crossmodal_similarity()
