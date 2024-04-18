"""Inspect outliers in the given dataset using the CLAP model and the proposed CCA method."""

import hydra
import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig

from mmda.utils.data_utils import (
    load_two_encoder_data,
    origin_centered,
)
from mmda.utils.dataset_utils import (
    filter_outliers,
    load_musiccaps,
)
from mmda.utils.sim_utils import (
    cosine_sim,
    weighted_corr_sim,
)


def inspect_youtube_data(cfg: DictConfig, sim_score: np.ndarray) -> None:
    """Inspect the YouTube data of the outliers.

    Args:
        cfg: Config dictionary.
        sim_score: similarity score.
        outlier_idx: index of the outliers.

    Returns:
        None
    """
    outlier_idx = filter_outliers(scores=sim_score, threshold=0.0)
    print(f"Number of outliers: {len(outlier_idx)}")
    print(f"Outlier index: {outlier_idx[:10]}")
    print(f"Scores of outliers: {sim_score[outlier_idx[:10]]}")
    dataframe = load_musiccaps(cfg)
    assert len(dataframe) == len(
        sim_score
    ), f"Dataframe length {len(dataframe)} != sim_score length {len(sim_score)}"

    for index in outlier_idx:
        row = dataframe.iloc[index]
        youtube_id = row["ytid"]
        start_s = row["start_s"]
        youtube_link = f"https://www.youtube.com/watch?v={youtube_id}&t={start_s}s"
        print(f"YouTube Link: \n{youtube_link}")
        print(f"data1 Path: \n{row['audio_path']}")
        print(f"Caption: \n{row['caption']}")
        print(f"Aspect List: \n{row['aspect_list']}")
        print(f"Start Time: \n{row['start_s']}")
        print(f"End Time: \n{row['end_s']}")
        input()


def clip_like_model_inspect_outliers(
    cfg: DictConfig, data1: np.ndarray, data2: np.ndarray
) -> None:
    """Inspect outliers in MusicCaps dataset using the CLAP model.

    Args:
        cfg: Config dictionary.
        data1: shape (N, D1).
        data2: shape (N, D2).

    Returns:
        None
    """
    ### Unsupervised data, so no labels of misaligned. Using all data.
    # calculate the similarity score
    sim_score = cosine_sim(data1, data2)
    inspect_youtube_data(cfg, sim_score)


def cca_inspect_outliers(cfg: DictConfig, data1: np.ndarray, data2: np.ndarray) -> None:
    """Inspect outliers in MusicCaps dataset using the propsoed CCA method.

    Args:
        cfg: Config dictionary.
        data1: shape (N, D1).
        data2: shape (N, D2).
    """
    ### Unsupervised data, so no labels of misaligned. Using all data.
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
    audio_text_cca = CCA(latent_dimensions=cfg.CCA_dim)
    data1, data2 = audio_text_cca.fit_transform((data1, data2))
    corr_align = (
        np.ones((data2.shape[1],))
        if cfg.equal_weights
        else np.diag(data1.T @ data2) / data1.shape[0]
    )
    # calculate the similarity score
    sim_score = weighted_corr_sim(data1, data2, corr_align, dim=cfg.sim_dim)
    inspect_youtube_data(cfg, sim_score)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Inspect outliers in the given dataset using the CLAP model and the proposed CCA method.

    Args:
        cfg: Config dictionary.

    Returns:
        None
    """
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    clip_like_model_inspect_outliers(cfg_dataset, data1, data2)
    cca_inspect_outliers(cfg_dataset, data1, data2)


if __name__ == "__main__":
    main()
