"""This module contains the code for the linear encoder experiment."""

import numpy as np
from omegaconf import DictConfig

from mmda.utils.linear_utils import (
    generate_gaussian,
    generate_transform_matrix,
    get_encoder,
    minimum_singular_value,
    solve_cca,
)


def linear_exps(cfg: DictConfig) -> None:
    """Main function for the linear encoder experiment.

    Args:
        cfg: config file
    """
    np.random.seed(cfg.seed)  # set the seed for reproducibility
    latent = generate_gaussian(
        cfg.num_data,
        np.zeros(cfg.latent_dim),
        np.eye(cfg.latent_dim),
    )
    noise1 = generate_gaussian(
        cfg.num_data,
        np.zeros(cfg.data_dim1),
        np.eye(cfg.data_dim1),
    )
    noise2 = generate_gaussian(
        cfg.num_data,
        np.zeros(cfg.data_dim2),
        np.eye(cfg.data_dim2),
    )
    transform1 = generate_transform_matrix((cfg.data_dim1, cfg.latent_dim))
    transform2 = generate_transform_matrix((cfg.data_dim2, cfg.latent_dim))
    data1 = transform1 @ latent + noise1
    data2 = transform2 @ latent + noise2
    enc1 = get_encoder(data1, cfg.latent_dim)
    enc2 = get_encoder(data2, cfg.latent_dim)

    z1 = enc1 @ data1
    z2 = enc2 @ data2
    # the analytical solution of CCA
    rho, cca_matrix1, cca_matrix2 = solve_cca(z1, z2)
    print(f"Correlation: {rho}")

    for ss in range(1, cfg.latent_dim):
        print(f"Latent dimension: {ss}")
        # calculate the SNR
        snr1 = np.linalg.norm(
            (cca_matrix1 @ enc1 @ transform1 @ latent)[:ss, :]
        ) / np.linalg.norm((cca_matrix1 @ enc1 @ noise1)[:ss, :])
        snr2 = np.linalg.norm(
            (cca_matrix2 @ enc2 @ transform2 @ latent)[:ss, :]
        ) / np.linalg.norm((cca_matrix2 @ enc2 @ noise2)[:ss, :])
        avg_snr = (snr1 + snr2) / 2
        avg_snr_db = 10 * np.log10(avg_snr)
        print(f"Average SNR: {avg_snr_db:.4f} dB")

        # average distance between data
        lambda1 = minimum_singular_value(cca_matrix1, ss) * minimum_singular_value(
            enc1, ss
        )
        lambda2 = minimum_singular_value(cca_matrix2, ss) * minimum_singular_value(
            enc2, ss
        )
        avg_lambda = (lambda1 + lambda2) / 2
        avg_lambda_db = 10 * np.log10(avg_lambda)
        print(f"Average lambda: {avg_lambda_db:.4f} dB")

        #
