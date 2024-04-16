import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig

import hydra
from mmda.utils.data_utils import (
    load_two_encoder_data,
    origin_centered,
)
from mmda.utils.dataset_utils import (
    filter_outliers,
    load_MusicCaps,
)
from mmda.utils.sim_utils import (
    cosine_sim,
    weighted_corr_sim,
)


def inspect_youtube_data(cfg: DictConfig, sim_score: np.ndarray):
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
    dataframe = load_MusicCaps(cfg)
    assert len(dataframe) == len(sim_score), f"Dataframe length {len(dataframe)} != sim_score length {len(sim_score)}"

    for index in outlier_idx:
        row = dataframe.iloc[index]
        YouTubeID = row["ytid"]
        start_s = row["start_s"]
        YouTubeLink = f"https://www.youtube.com/watch?v={YouTubeID}&t={start_s}s"
        print(f"YouTube Link: \n{YouTubeLink}")
        print(f"Data1 Path: \n{row['audio_path']}")
        print(f"Caption: \n{row['caption']}")
        print(f"Aspect List: \n{row['aspect_list']}")
        print(f"Start Time: \n{row['start_s']}")
        print(f"End Time: \n{row['end_s']}")
        input()


def CLIP_like_model_inspect_outliers(cfg: DictConfig, Data1: np.ndarray, Data2: np.ndarray):
    """Inspect outliers in MusicCaps dataset using the CLAP model.

    Args:
        cfg: Config dictionary.
        Data1: shape (N, D1).
        Data2: shape (N, D2).

    Returns:
        None
    """
    ### Unsupervised data, so no labels of misaligned. Using all data.
    # calculate the similarity score
    sim_score = cosine_sim(Data1, Data2)
    inspect_youtube_data(cfg, sim_score)


def CCA_inspect_outliers(cfg: DictConfig, Data1: np.ndarray, Data2: np.ndarray):
    """Inspect outliers in MusicCaps dataset using the propsoed CCA method.

    Args:
        cfg: Config dictionary.
        Data1: shape (N, D1).
        Data2: shape (N, D2).
    """
    ### Unsupervised data, so no labels of misaligned. Using all data.
    # zero mean data
    Data1, _ = origin_centered(Data1)
    Data2, _ = origin_centered(Data2)
    # make sure the data is zero mean
    assert np.allclose(Data1.mean(axis=0), 0, atol=1e-4), f"Data1 not zero mean: {Data1.mean(axis=0)}"
    assert np.allclose(Data2.mean(axis=0), 0, atol=1e-4), f"Data2 not zero mean: {Data2.mean(axis=0)}"

    # CCA dimensionality reduction
    audio_text_CCA = CCA(latent_dimensions=cfg.CCA_dim)
    Data1, Data2 = audio_text_CCA.fit_transform((Data1, Data2))
    if cfg.equal_weights:
        corr_align = np.ones((Data2.shape[1],))  # dim,
    else:
        corr_align = np.diag(Data1.T @ Data2) / Data1.shape[0]  # dim,

    # calculate the similarity score
    sim_score = weighted_corr_sim(Data1, Data2, corr_align, dim=cfg.sim_dim)
    inspect_youtube_data(cfg, sim_score)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Inspect outliers in the given dataset using the CLAP model and the proposed CCA method.

    Args:
        cfg: Config dictionary.

    Returns:
        None
    """
    cfg_dataset, Data1, Data2 = load_two_encoder_data(cfg)
    CLIP_like_model_inspect_outliers(cfg_dataset, Data1, Data2)
    CCA_inspect_outliers(cfg_dataset, Data1, Data2)


if __name__ == "__main__":
    main()
