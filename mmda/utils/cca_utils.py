"""Canonical Correlation Analysis (CCA) related functions."""

import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig


def cca_fit_train_data(
    cfg_dataset: DictConfig, traindata1: np.ndarray, traindata2: np.ndarray
) -> tuple[CCA, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the CCA model to the training data.

    Args:
        cfg_dataset: the dataset configuration
        traindata1: the first training data. shape: (num_samples, dim)
        traindata2: the second training data. shape: (num_samples, dim)

    Returns:
        cca: the CCA model
        traindata1: the first training data after CCA. shape: (num_samples, dim)
        traindata2: the second training data after CCA. shape: (num_samples, dim)
        corr_align: the correlation alignment. shape: (dim,)
    """
    # Check the shape of the training data9
    assert (
        traindata1.shape[0] >= traindata1.shape[1]
    ), f"The number of samples {traindata1.shape[0]} should be larger than features {traindata1.shape[1]}"
    assert (
        traindata2.shape[0] >= traindata2.shape[1]
    ), f"The number of samples {traindata2.shape[0]} should be larger than features {traindata2.shape[1]}"

    # CCA dimensionality reduction
    cca = CCA(latent_dimensions=cfg_dataset.sim_dim)
    traindata1, traindata2 = cca.fit_transform((traindata1, traindata2))
    if cfg_dataset.equal_weights:
        corr_align = np.ones((traindata2.shape[1],))  # dim,
    else:
        corr_align = np.diag(traindata1.T @ traindata2) / traindata1.shape[0]  # dim,
    return cca, traindata1, traindata2, corr_align
