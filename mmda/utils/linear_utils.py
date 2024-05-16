"""This module contains utility functions for linear algebra and linear setting operations."""

import numpy as np


def svd(matrix: np.ndarray) -> np.ndarray:
    """Compute the singular value decomposition of a matrix.

    Args:
        matrix: the matrix to be decomposed

    Returns:
        np.ndarray: the singular value decomposition of the matrix
    """
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    assert np.allclose(np.eye(matrix.shape[0]), u.T @ u)
    assert np.allclose(np.eye(matrix.shape[1]), vh @ vh.T)
    return u, s, vh


def minimum_singular_value(matrix: np.ndarray, dim: int | None = None) -> np.ndarray:
    """Compute the minimum singular value of a matrix.

    Args:
        matrix: the matrix
        dim: the dimension of the singular value. 1-based index
    Returns:
        np.ndarray: the minimum singular value of the matrix
    """
    _, s, _ = svd(matrix)
    return s[dim - 1] if dim is not None else s[-1]


def generate_gaussian(size: int, mean: float, std: float) -> np.ndarray:
    """Generate a Gaussian distribution.

    Args:
        size: the shape of the Gaussian distribution
        mean: the mean of the Gaussian distribution
        std: the standard deviation of the Gaussian distribution

    Returns:
        np.ndarray: the Gaussian distribution
    """
    gauss = np.random.multivariate_normal(mean, std, size).T
    assert gauss.shape == (mean.shape[0], size)
    return gauss


def generate_transform_matrix(shape: int) -> np.ndarray:
    """Generate a transformation matrix.

    Args:
        shape: the shape of the transformation matrix

    Returns:
        np.ndarray: the transformation matrix. shape: (data_dim, latent_dim)
    """
    return np.random.rand(*shape)


def get_encoder(data: np.ndarray, latent_dim: int) -> np.ndarray:
    """Get the encoder of the data.

    Args:
        data: the data to be encoded
        latent_dim: the latent dimension of the data

    Returns:
        enc: np.ndarray: the encoder of the data. shape: (latent_dim, data_dim)
    """

    def off_diag(matrix: np.ndarray) -> np.ndarray:
        """Remain only the off-diagonal elements of a matrix.

        Args:
            matrix: the matrix matrix

        Returns:
            np.ndarray: the off-diagonal elements of the matrix
        """
        offdiag = matrix - np.diag(np.diag(matrix))
        assert offdiag.shape == matrix.shape
        assert (np.diag(offdiag) >= 0).all()
        return offdiag

    num_data = data.shape[1]
    data_off_diag = off_diag(data @ data.T)
    to_be_svd = (
        data_off_diag
        - 1 / (num_data - 1) * data @ np.ones((num_data, num_data)) @ data.T
    )
    u, s, vh = svd(to_be_svd)
    enc = np.zeros((latent_dim, data.shape[0]))
    normal_standard_basis = np.eye(latent_dim)
    for i in range(latent_dim):
        enc += (
            u[:, i : i + 1] @ s[i].reshape(1, 1) @ normal_standard_basis[i : i + 1, :]
        ).T
    return enc / num_data


def solve_cca(
    data1: np.ndarray, data2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the CCA problem.

    Args:
        data1: the first data
        data2: the second data

    Returns:
        np.ndarray: the first CCA transform matrix
        np.ndarray: the second CCA transform matrix
        np.ndarray: the correlation coefficient
    """

    def half_matrix(matrix: np.ndarray) -> np.ndarray:
        """Remain only the half of the matrix.

        Args:
            matrix: a PSD matrix

        Returns:
            np.ndarray: the half of the matrix
        """
        # Computing diagonalization
        evalues, evectors = np.linalg.eigh(matrix)
        assert (evalues >= 0).all()
        return evectors * np.sqrt(evalues) @ evectors.T

    cov12 = data1 @ data2.T
    cov11 = data1 @ data1.T
    cov22 = data2 @ data2.T
    cov11_sqrt_inv = np.linalg.inv(half_matrix(cov11))
    cov22_sqrt_inv = np.linalg.inv(half_matrix(cov22))
    u, s, vh = svd(cov11_sqrt_inv @ cov12 @ cov22_sqrt_inv)
    cca_matrix1 = u.T @ cov11_sqrt_inv
    cca_matrix2 = vh @ cov22_sqrt_inv
    return s, cca_matrix1, cca_matrix2
