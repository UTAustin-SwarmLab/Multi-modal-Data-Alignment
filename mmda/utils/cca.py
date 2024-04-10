import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig


def CCA_fit_train_data(
    cfg_dataset: DictConfig, trainData1: np.ndarray, trainData2: np.ndarray
) -> tuple[CCA, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the CCA model to the training data.

    Args:
        cfg_dataset: the dataset configuration
        trainData1: the first training data. shape: (num_samples, dim)
        trainData2: the second training data. shape: (num_samples, dim)

    Returns:
        cca: the CCA model
        trainData1: the first training data after CCA. shape: (num_samples, dim)
        trainData2: the second training data after CCA. shape: (num_samples, dim)
        corr_align: the correlation alignment. shape: (dim,)
    """
    # CCA dimensionality reduction
    assert trainData1.shape[0] >= trainData1.shape[1], "The number of samples should be larger than features"
    assert trainData2.shape[0] >= trainData2.shape[1], "The number of samples should be larger than features"
    cca = CCA(latent_dimensions=cfg_dataset.CCA_dim)
    trainData1, trainData2 = cca.fit_transform((trainData1, trainData2))
    if cfg_dataset.equal_weights:
        corr_align = np.ones((trainData2.shape[1],))  # dim,
    else:
        corr_align = np.diag(trainData1.T @ trainData2) / trainData1.shape[0]  # dim,
    assert np.max(corr_align) <= 1.0, f"max corr_align: {np.max(corr_align)}"
    assert np.min(corr_align) >= 0.0, f"min corr_align: {np.min(corr_align)}"
    return cca, trainData1, trainData2, corr_align
