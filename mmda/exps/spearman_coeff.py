"""calculate the spearman's rank coefficient with CLIP model's and our proposed method's similarity score."""

import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig

from mmda.utils.data_utils import (
    load_clip_like_data,
    load_two_encoder_data,
    origin_centered,
)
from mmda.utils.sim_utils import (
    cosine_sim,
    spearman_rank_coefficient,
    spearman_to_p_value,
    weighted_corr_sim,
)


def cal_spearman_coeff(
    cfg: DictConfig,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the Spearman's rank coeff of MusicCaps with CLIP model and CCA similarity score.

    Args:
        cfg: Config dictionary.

    Returns:
        r: Spearman's rank coefficient.
        p_value: p-value.
        sim_score_clip: CLIP similarity score.
        sim_score_CCA: CCA similarity score.
        rank_clip: Rank of CLIP similarity score.
        rank_cca: Rank of CCA similarity score.
    """
    ### Unsupervised data, so no labels of misaligned. Using all data.
    # calculate the similarity score of CLIP
    cfg_dataset, clip_data1, clip_data2 = load_clip_like_data(cfg)
    sim_score_clip = cosine_sim(clip_data1, clip_data2)

    # load embeddings from the two encoders
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)

    # zero mean data
    data1, _ = origin_centered(data1)
    data2, _ = origin_centered(data2)
    # make sure the data is zero mean
    assert np.allclose(
        data1.mean(axis=0), 0, atol=1e-4
    ), f"data1 not zero mean: {data1.mean(axis=0)}"
    assert np.allclose(
        data2.mean(axis=0), 0, atol=1e-4
    ), f"data2 not zero mean: {data2.mean(axis=0)}"

    # CCA dimensionality reduction
    audio_text_cca = CCA(latent_dimensions=cfg_dataset.sim_dim)
    data1, data2 = audio_text_cca.fit_transform((data1, data2))
    corr_align = (
        np.ones((data2.shape[1],))
        if cfg_dataset.equal_weights
        else np.diag(data1.T @ data2) / data1.shape[0]
    )

    # calculate the similarity score
    sim_score_cca = weighted_corr_sim(data1, data2, corr_align, dim=cfg_dataset.sim_dim)

    r, rank_clip, rank_cca = spearman_rank_coefficient(sim_score_clip, sim_score_cca)
    p_value = spearman_to_p_value(r, len(sim_score_clip))

    return r, p_value, sim_score_clip, sim_score_cca, rank_clip, rank_cca
