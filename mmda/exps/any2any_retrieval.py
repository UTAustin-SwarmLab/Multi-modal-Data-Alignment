"""This module contains the functions to detect mislabeled data using the proposed method and baselines."""

import numpy as np
from omegaconf import DictConfig

from mmda.utils.retrieval_dataset_class import load_retrieval_dataset


def any2any_retrieval(cfg: DictConfig) -> tuple[list, list, list]:
    """Retrieve the multimodal datasets using the any2any conformal retrieval.

    Args:
        cfg: configuration file

    Returns:
        mAPs: mean average precision scores at 5 and 20
        precisions: precision at 1, 5, 20
        recalls: recall at 1, 5, 20
    """
    ds = load_retrieval_dataset(cfg)
    ds.preprocess_retrieval_data()
    ds.train_crossmodal_similarity()
    ds.get_cali_data()
    ds.get_test_data()
    ds.cal_test_conformal_prob()
    single5_aps, single5_precisions, single5_recalls = ds.retrieve_data(mode="single")
    aps, precisions, recalls = ds.retrieve_data(mode="miss")
    full_aps, full_precisions, full_recalls = ds.retrieve_data(mode="full")
    return (
        (
            {5: np.mean(aps[5]), 20: np.mean(aps[20])},
            {
                1: np.mean(precisions[1]),
                5: np.mean(precisions[5]),
                20: np.mean(precisions[20]),
            },
            {
                1: np.mean(recalls[1]),
                5: np.mean(recalls[5]),
                20: np.mean(recalls[20]),
            },
        ),
        (
            {5: np.mean(full_aps[5]), 20: np.mean(full_aps[20])},
            {
                1: np.mean(full_precisions[1]),
                5: np.mean(full_precisions[5]),
                20: np.mean(full_precisions[20]),
            },
            {
                1: np.mean(full_recalls[1]),
                5: np.mean(full_recalls[5]),
                20: np.mean(full_recalls[20]),
            },
        ),
        (
            single5_aps,
            single5_precisions,
            single5_recalls,
        ),
    )
