"""This module contains utility functions for loading data feature embeddings."""

from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig

from mmda.utils.dataset_utils import load_dataset_config


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
    cfg_dataset = load_dataset_config(cfg)
    # load image/audio embeddings and text embeddings
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
    cfg_dataset = load_dataset_config(cfg)
    # load image/audio embeddings and text embeddings
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
            Path(cfg_dataset.paths.save_path + "ImageNet_img_emb_clip.pkl")
        )
        data2 = joblib.load(
            Path(cfg_dataset.paths.save_path + "ImageNet_text_emb_clip.pkl")
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
