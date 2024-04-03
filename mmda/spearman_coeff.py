import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig

from mmda.utils.data_utils import (
    load_CLIP_like_data,
    load_two_encoder_data,
    origin_centered,
)
from mmda.utils.sim_utils import (
    Spearman_rank_coefficient,
    Spearman_to_p_value,
    cosine_sim,
    weighted_corr_sim,
)


def cal_spearman_coeff(cfg: DictConfig) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the Spearman's rank coeff of MusicCaps with CLIP model and CCA similarity score.

    Args:
        cfg: Config dictionary.

    Returns:
        r: Spearman's rank coefficient.
        p_value: p-value.
        sim_score_CLIP: CLIP similarity score.
        sim_score_CCA: CCA similarity score.
        rankCLIP: Rank of CLIP similarity score.
        rankCCA: Rank of CCA similarity score.
    """
    ### Unsupervised data, so no labels of misaligned. Using all data.
    # calculate the similarity score of CLIP
    cfg_dataset, CLIP_data1, CLIP_data2 = load_CLIP_like_data(cfg)
    sim_score_CLIP = cosine_sim(CLIP_data1, CLIP_data2)

    # load embeddings from the two encoders
    cfg_dataset, Data1, Data2 = load_two_encoder_data(cfg)

    # zero mean data
    Data1, _ = origin_centered(Data1)
    Data2, _ = origin_centered(Data2)
    # make sure the data is zero mean
    assert np.allclose(Data1.mean(axis=0), 0, atol=1e-4), f"Data1 not zero mean: {Data1.mean(axis=0)}"
    assert np.allclose(Data2.mean(axis=0), 0, atol=1e-4), f"Data2 not zero mean: {Data2.mean(axis=0)}"

    # CCA dimensionality reduction
    audio_text_CCA = CCA(latent_dimensions=cfg_dataset.CCA_dim)
    Data1, Data2 = audio_text_CCA.fit_transform((Data1, Data2))
    if cfg_dataset.equal_weights:
        corr_align = np.ones((Data2.shape[1],))  # dim,
    else:
        corr_align = np.diag(Data1.T @ Data2) / Data1.shape[0]  # dim,

    # calculate the similarity score
    sim_score_CCA = weighted_corr_sim(Data1, Data2, corr_align, dim=cfg_dataset.sim_dim)

    r, rankCLIP, rankCCA = Spearman_rank_coefficient(sim_score_CLIP, sim_score_CCA)
    p_value = Spearman_to_p_value(r, len(sim_score_CLIP))

    return r, p_value, sim_score_CLIP, sim_score_CCA, rankCLIP, rankCCA
