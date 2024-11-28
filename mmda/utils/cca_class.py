"""Canonical Correlation Analysis (CCA) related functions."""

import pickle
from pathlib import Path

import joblib
import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig
from scipy.linalg import sqrtm

from mmda.utils.data_utils import origin_centered


class NormalizedCCA:
    """Canonical Correlation Analysis (CCA) class which automatically zero-mean data."""

    def __init__(self, sim_dim: int | None = None) -> None:
        """Initialize the CCA model."""
        self.traindata1_mean = None
        self.traindata2_mean = None
        self.sim_dim = sim_dim

    def fit_transform_train_data(
        self, cfg_dataset: DictConfig, traindata1: np.ndarray, traindata2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the CCA model to the training data.

        Args:
            cfg_dataset: the dataset configuration
            traindata1: the first training data. shape: (num_samples, dim)
            traindata2: the second training data. shape: (num_samples, dim)

        Returns:
            traindata1: the first training data after CCA. shape: (num_samples, dim)
            traindata2: the second training data after CCA. shape: (num_samples, dim)
            corr_coeff: the correlation coefficient. shape: (dim,)
        """
        # Check the shape of the training data
        # zero mean data
        traindata1, traindata1_mean = origin_centered(traindata1)
        traindata2, traindata2_mean = origin_centered(traindata2)
        self.traindata1_mean, self.traindata2_mean = traindata1_mean, traindata2_mean
        self.traindata1, self.traindata2 = traindata1, traindata2

        # check if training data is zero-mean
        assert np.allclose(
            traindata1.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        ), f"traindata1align not zero mean: {max(abs(traindata1.mean(axis=0)))}"
        assert np.allclose(
            traindata2.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        ), f"traindata2align not zero mean: {max(abs(traindata2.mean(axis=0)))}"

        # CCA dimensionality reduction
        if self.sim_dim is None:
            self.cca = CCA(latent_dimensions=cfg_dataset.sim_dim)
        else:
            self.cca = CCA(latent_dimensions=self.sim_dim)
        traindata1, traindata2 = self.cca.fit_transform((traindata1, traindata2))
        if cfg_dataset.equal_weights:
            corr_coeff = np.ones((traindata2.shape[1],))  # dim,
        else:
            corr_coeff = (
                np.diag(traindata1.T @ traindata2) / traindata1.shape[0]
            )  # dim,
        assert (
            corr_coeff >= 0
        ).all(), f"Correlation should be non-negative. {corr_coeff}"
        assert (
            corr_coeff <= 1
        ).all(), f"Correlation should be less than 1. {corr_coeff}"
        self.corr_coeff = corr_coeff
        self.traindata1, self.traindata2 = traindata1, traindata2
        return traindata1, traindata2, corr_coeff

    def transform_data(
        self, data1: tuple[np.ndarray, np.ndarray], data2: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform the data using the fitted CCA model.

        Args:
            data1: the first data. shape: (num_samples, dim)
            data2: the second data. shape: (num_samples, dim)

        Returns:
            data1: the first transformed data. shape: (num_samples, dim)
            data2: the second transformed data. shape: (num_samples, dim)
        """
        assert self.traindata1_mean is not None, "Please fit the cca model first."
        assert self.traindata2_mean is not None, "Please fit the cca model first."
        # zero mean data and transform
        data1 = data1 - self.traindata1_mean
        data2 = data2 - self.traindata2_mean
        data1, data2 = self.cca.transform((data1, data2))
        return data1, data2

    def save_model(self, path: str | Path) -> None:
        """Save the CCA class.

        Args:
            path: the path to save the class
        """
        if isinstance(path, str):
            path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def load_model(self, path: str | Path) -> None:
        """Load the CCA class.

        Args:
            path: the path to load the class
        """
        if isinstance(path, str):
            path = Path(path)
        self.__dict__ = joblib.load(path.open("rb")).__dict__


class ReNormalizedCCA:
    """Canonical Correlation Analysis (CCA) class which automatically zero-mean data."""

    def __init__(self, sim_dim: int | None = None) -> None:
        """Initialize the CCA model."""
        self.traindata1_mean = None
        self.traindata2_mean = None
        self.sim_dim = sim_dim

    def fit_transform_train_data(
        self, cfg_dataset: DictConfig, traindata1: np.ndarray, traindata2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the CCA model to the training data.

        Args:
            cfg_dataset: the dataset configuration
            traindata1: the first training data. shape: (num_samples, dim)
            traindata2: the second training data. shape: (num_samples, dim)

        Returns:
            traindata1: the first training data after CCA. shape: (num_samples, dim)
            traindata2: the second training data after CCA. shape: (num_samples, dim)
            corr_coeff: the correlation coefficient. shape: (dim,)
        """
        # Check the shape of the training data
        # zero mean data
        traindata1, traindata1_mean = origin_centered(traindata1)
        traindata2, traindata2_mean = origin_centered(traindata2)
        self.traindata1_mean, self.traindata2_mean = traindata1_mean, traindata2_mean

        # check if training data is zero-mean
        assert np.allclose(
            traindata1.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        ), f"traindata1align not zero mean: {max(abs(traindata1.mean(axis=0)))}"
        assert np.allclose(
            traindata2.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        ), f"traindata2align not zero mean: {max(abs(traindata2.mean(axis=0)))}"

        # CCA dimensionality reduction
        print((traindata1.T @ traindata1).shape)
        sigma_z1_inv = np.linalg.inv(traindata1.T @ traindata1)
        sigma_z1_inv_sqrt = sqrtm(sigma_z1_inv)
        assert np.allclose(
            sigma_z1_inv_sqrt @ sigma_z1_inv_sqrt, sigma_z1_inv
        ), "sigma_z1_inv_sqrt is not the square root of sigma_z1_inv"
        sigma_z2_inv = np.linalg.inv(traindata2.T @ traindata2)
        sigma_z2_inv_sqrt = sqrtm(sigma_z2_inv)
        assert np.allclose(
            sigma_z2_inv_sqrt @ sigma_z2_inv_sqrt, sigma_z2_inv
        ), "sigma_z2_inv_sqrt is not the square root of sigma_z2_inv"

        svd_mat = sigma_z1_inv_sqrt @ traindata1.T @ traindata2 @ sigma_z2_inv_sqrt
        u, s, vh = np.linalg.svd(svd_mat)
        assert np.allclose(
            u @ np.diag(s) @ vh, svd_mat
        ), "svd_mat is not the SVD of svd_mat"

        self.A = u @ sigma_z1_inv_sqrt
        self.B = vh @ sigma_z2_inv_sqrt

        corr_coeff = np.ones((traindata2.shape[1],)) if cfg_dataset.equal_weights else s
        assert (
            corr_coeff >= 0
        ).all(), f"Correlation should be non-negative. {corr_coeff}"
        assert (
            corr_coeff <= 1
        ).all(), f"Correlation should be less than 1. {corr_coeff}"
        self.corr_coeff = corr_coeff
        self.traindata1, self.traindata2 = (
            (self.A @ traindata1.T).T,
            (self.B @ traindata2.T).T,
        )
        return self.traindata1, self.traindata2, corr_coeff

    def transform_data(
        self, data1: tuple[np.ndarray, np.ndarray], data2: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform the data using the fitted CCA model.

        Args:
            data1: the first data. shape: (num_samples, dim)
            data2: the second data. shape: (num_samples, dim)

        Returns:
            data1: the first transformed data. shape: (num_samples, dim)
            data2: the second transformed data. shape: (num_samples, dim)
        """
        assert self.traindata1_mean is not None, "Please fit the cca model first."
        assert self.traindata2_mean is not None, "Please fit the cca model first."
        # zero mean data and transform
        data1 = data1 - self.traindata1_mean
        data2 = data2 - self.traindata2_mean
        data1, data2 = (self.A @ data1.T).T, (self.B @ data2.T).T
        return data1, data2

    def save_model(self, path: str | Path) -> None:
        """Save the CCA class.

        Args:
            path: the path to save the class
        """
        if isinstance(path, str):
            path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def load_model(self, path: str | Path) -> None:
        """Load the CCA class.

        Args:
            path: the path to load the class
        """
        if isinstance(path, str):
            path = Path(path)
        self.__dict__ = joblib.load(path.open("rb")).__dict__
