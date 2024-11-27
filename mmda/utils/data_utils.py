"""This module contains utility functions for loading data feature embeddings."""

from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig

import hydra


def load_three_encoder_data(
    cfg: DictConfig,
) -> tuple[DictConfig, np.ndarray, np.ndarray, np.ndarray]:
    """Load the data in three modalities.

    Args:
        cfg: configuration file
    Returns:
        cfg_dataset: configuration file for the dataset
        data1: data in modality image. shape: (N, D1)
        data2: data in modality lidar. shape: (N, D2)
        data3: data in modality text. shape: (N, D3)
    """
    dataset = cfg.dataset
    cfg_dataset = cfg[cfg.dataset]
    # load image & lidar & text embeddings
    if dataset == "KITTI":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"KITTI_camera_emb_{cfg_dataset.img_encoder}.pkl"
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"KITTI_lidar_emb_{cfg_dataset.lidar_encoder}.pkl"
            )
        )
        data3 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"KITTI_text_emb_{cfg_dataset.text_encoder}.pkl"
            )
        )
    # TODO: add more datasets
    else:
        msg = f"Dataset {dataset} not supported."
        raise ValueError(msg)
    return cfg_dataset, data1, data2, data3


def load_two_encoder_data(cfg: DictConfig) -> tuple[DictConfig, np.ndarray, np.ndarray]:
    """Load the data in two modalities.

    Args:
        cfg: configuration file
    Returns:
        cfg_dataset: configuration file for the dataset
        data1: data in modality 1. shape: (N, D1)
        data2: data in modality 2. shape: (N, D2)
    """
    dataset = cfg.dataset
    cfg_dataset = cfg[cfg.dataset]
    # load image/audio embeddings and text embeddings
    if dataset == "sop":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"data/SOP_img_emb_{cfg_dataset.img_encoder}.pkl"
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"data/SOP_text_emb_{cfg_dataset.text_encoder}.pkl"
            )
        )
    elif dataset == "musiccaps":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"MusicCaps_audio_emb_{cfg_dataset.img_encoder}.pkl",
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"MusicCaps_text_emb_{cfg_dataset.text_encoder}.pkl",
            )
        )
    elif dataset == "imagenet":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"ImageNet_img_emb_{cfg_dataset.img_encoder}.pkl",
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"ImageNet_text_emb_{cfg_dataset.text_encoder}.pkl",
            )
        )
    elif dataset == "tiil":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"TIIL_img_emb_{cfg_dataset.img_encoder}.pkl"
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"TIIL_text_emb_{cfg_dataset.text_encoder}.pkl",
            )
        )
    elif dataset == "cosmos":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"COSMOS_img_emb_{cfg_dataset.img_encoder}.pkl",
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"COSMOS_text_emb_{cfg_dataset.text_encoder}.pkl",
            )
        )
    elif dataset == "pitts":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"PITTS_img_emb_{cfg_dataset.img_encoder}.pkl",
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"PITTS_text_emb_{cfg_dataset.text_encoder}.pkl",
            )
        )
    elif dataset == "flickr":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"FLICKR_img_emb_{cfg_dataset.img_encoder}.pkl",
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"FLICKR_text_emb_{cfg_dataset.text_encoder}.pkl",
            )
        )
    elif dataset == "leafy_spurge":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"LeafySpurge_img_emb_{cfg_dataset.img_encoder}.pkl",
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"LeafySpurge_text_emb_{cfg_dataset.text_encoder}.pkl",
            )
        )
    # TODO: add more datasets
    else:
        msg = f"Dataset {dataset} not supported."
        raise ValueError(msg)
    return cfg_dataset, data1, data2


def load_clip_like_data(cfg: DictConfig) -> tuple[DictConfig, np.ndarray, np.ndarray]:
    """Load the data in two modalities. The encoders are the same CLIP like model.

    Args:
        cfg: configuration file
    Returns:
        cfg_dataset: configuration file for the dataset
        data1: data in modality 1. shape: (N, D1)
        data2: data in modality 2. shape: (N, D2)
    """
    dataset = cfg.dataset
    cfg_dataset = cfg[cfg.dataset]
    # load image/audio embeddings and text embeddings
    if dataset == "sop":
        data1 = joblib.load(
            Path(cfg_dataset.paths.save_path + "data/SOP_img_emb_clip.pkl")
        )
        data2 = joblib.load(
            Path(cfg_dataset.paths.save_path + "data/SOP_text_emb_clip.pkl")
        )
    elif dataset == "musiccaps":
        data1 = joblib.load(
            Path(cfg_dataset.paths.save_path + "MusicCaps_audio_emb_clap.pkl")
        )
        data2 = joblib.load(
            Path(cfg_dataset.paths.save_path + "MusicCaps_text_emb_clap.pkl")
        )
    elif dataset == "imagenet":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"ImageNet_img_emb_clip{cfg_dataset.model_name}.pkl"
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"ImageNet_text_emb_clip{cfg_dataset.model_name}.pkl"
            )
        )
    elif dataset == "tiil":
        data1 = joblib.load(Path(cfg_dataset.paths.save_path + "TIIL_img_emb_clip.pkl"))
        data2 = joblib.load(
            Path(cfg_dataset.paths.save_path + "TIIL_text_emb_clip.pkl")
        )
    elif dataset == "cosmos":
        data1 = joblib.load(
            Path(cfg_dataset.paths.save_path + "COSMOS_img_emb_clip.pkl")
        )
        data2 = joblib.load(
            Path(cfg_dataset.paths.save_path + "COSMOS_text_emb_clip.pkl")
        )
    elif dataset == "pitts":
        data1 = joblib.load(
            Path(cfg_dataset.paths.save_path + "PITTS_img_emb_clip.pkl")
        )
        data2 = joblib.load(
            Path(cfg_dataset.paths.save_path + "PITTS_text_emb_clip.pkl")
        )
    elif dataset == "flickr":
        data1 = joblib.load(
            Path(cfg_dataset.paths.save_path + "FLICKR_img_emb_clip.pkl")
        )
        data2 = joblib.load(
            Path(cfg_dataset.paths.save_path + "FLICKR_text_emb_clip.pkl")
        )
    elif dataset == "leafy_spurge":
        data1 = joblib.load(
            Path(
                cfg_dataset.paths.save_path + "LeafySpurge_img_emb_clip.pkl",
            )
        )
        data2 = joblib.load(
            Path(
                cfg_dataset.paths.save_path + "LeafySpurge_text_emb_clip.pkl",
            )
        )
    # TODO: add more datasets
    else:
        msg = f"Dataset {dataset} not supported."
        raise ValueError(msg)
    return cfg_dataset, data1, data2


def origin_centered(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """This function returns the origin centered data matrix and the mean of each feature.

    Args:
        x: data matrix (n_samples, n_features)

    Returns:
        origin centered data matrix, mean of each feature
    """
    return x - np.mean(x, axis=0), np.mean(x, axis=0)


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def test(cfg: DictConfig) -> None:  # noqa: D103
    cfg_dataset, data1, data2, data3 = load_three_encoder_data(cfg)
    print(data1.shape)
    print(data2.shape)
    print(data3.shape)


if __name__ == "__main__":
    test()
