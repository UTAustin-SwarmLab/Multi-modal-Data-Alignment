"""calculate the spearman's rank coefficient with CLIP model's and our proposed method's similarity score."""

import numpy as np
from omegaconf import DictConfig

from mmda.utils.cca_class import NormalizedCCA
from mmda.utils.data_utils import (
    load_clip_like_data,
    load_two_encoder_data,
)
from mmda.utils.retrieval_dataset_class import load_retrieval_dataset
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

    # CCA dimensionality reduction
    cca = NormalizedCCA()
    data1, data2, corr_align = cca.fit_transform_train_data(cfg_dataset, data1, data2)

    # calculate the similarity score
    sim_score_cca = weighted_corr_sim(data1, data2, corr_align, dim=cfg_dataset.sim_dim)

    r, rank_clip, rank_cca = spearman_rank_coefficient(sim_score_clip, sim_score_cca)
    p_value = spearman_to_p_value(r, len(sim_score_clip))

    return r, p_value, sim_score_clip, sim_score_cca, rank_clip, rank_cca


def retrieval_spearman_coeff(cfg: DictConfig) -> tuple[float, float]:
    """Calculate the Spearman's rank coeff of MusicCaps with CLIP model and CCA similarity score for retrieval.

    Args:
        cfg: Config dictionary.

    Returns:
        avg_r: Averaged Spearman's rank coefficient of all data points.
        avg_p_value: Averaged p-value of all data points.
    """
    # calculate the similarity score of CLIP
    cfg_dataset, clip_data1, clip_data2 = load_clip_like_data(cfg)
    clip_ds = load_retrieval_dataset(cfg)
    clip_ds.preprocess_retrieval_data(clip_data1, clip_data2)
    sim_scores_clip = clip_ds.map_precision_similarity(sim_fn=cosine_sim)[-1]

    # load embeddings from the two encoders and run CCA
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    cca_ds = load_retrieval_dataset(cfg)
    cca_ds.preprocess_retrieval_data(data1, data2)

    # CCA dimensionality reduction
    cca = NormalizedCCA()
    cca_ds.traindata1, cca_ds.traindata2, corr_align = cca.fit_transform_train_data(
        cfg_dataset, cca_ds.traindata1, cca_ds.traindata2
    )
    cca_ds.testdata1, cca_ds.testdata2 = cca.transform_data(
        cca_ds.testdata1, cca_ds.testdata2
    )

    def sim_fn(x: np.array, y: np.array) -> np.array:
        return weighted_corr_sim(x, y, corr=corr_align, dim=cfg_dataset.sim_dim)

    sim_scores_cca = cca_ds.map_precision_similarity(sim_fn=sim_fn)[-1]

    assert (
        sim_scores_clip.shape == sim_scores_cca.shape
    ), f"{sim_scores_clip.shape} != {sim_scores_cca.shape}"
    r_list, p_value_list = [], []
    for i in range(sim_scores_clip.shape[0]):
        r, rank_clip, rank_cca = spearman_rank_coefficient(
            sim_scores_clip[i, :], sim_scores_cca[i, :]
        )
        p_value = spearman_to_p_value(r, sim_scores_clip.shape[1])
        r_list.append(r)
        p_value_list.append(p_value)
    avg_r = np.mean(r_list)
    avg_p_value = np.mean(p_value_list)

    return avg_r, avg_p_value
