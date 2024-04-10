import pickle

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
    if dataset == "sop":
        with open(cfg_dataset.paths.save_path + f"data/SOP_img_emb_{cfg_dataset.img_encoder}.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + f"data/SOP_text_emb_{cfg_dataset.text_encoder}.pkl", "rb") as f:
            Data2 = pickle.load(f)
    elif dataset == "musiccaps":
        with open(cfg_dataset.paths.save_path + f"MusicCaps_audio_emb_{cfg_dataset.audio_encoder}.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + f"MusicCaps_text_emb_{cfg_dataset.text_encoder}.pkl", "rb") as f:
            Data2 = pickle.load(f)
    elif dataset == "imagenet":
        with open(cfg_dataset.paths.save_path + f"ImageNet_img_emb_{cfg_dataset.img_encoder}.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + f"ImageNet_text_emb_{cfg_dataset.text_encoder}.pkl", "rb") as f:
            Data2 = pickle.load(f)
    elif dataset == "tiil":
        with open(cfg_dataset.paths.save_path + f"TIIL_img_emb_{cfg_dataset.img_encoder}.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + f"TIIL_text_emb_{cfg_dataset.text_encoder}.pkl", "rb") as f:
            Data2 = pickle.load(f)
    elif dataset == "cosmos":
        with open(cfg_dataset.paths.save_path + f"COSMOS_img_emb_{cfg_dataset.img_encoder}.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + f"COSMOS_text_emb_{cfg_dataset.text_encoder}.pkl", "rb") as f:
            Data2 = pickle.load(f)
    # TODO: add more datasets
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return cfg_dataset, Data1, Data2


def load_CLIP_like_data(cfg: DictConfig) -> tuple[DictConfig, np.ndarray, np.ndarray]:
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
    if dataset == "sop":
        with open(cfg_dataset.paths.save_path + "data/SOP_img_emb_clip.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + "data/SOP_text_emb_clip.pkl", "rb") as f:
            Data2 = pickle.load(f)
    elif dataset == "musiccaps":
        with open(cfg_dataset.paths.save_path + "MusicCaps_audio_emb_clap.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + "MusicCaps_text_emb_clap.pkl", "rb") as f:
            Data2 = pickle.load(f)
    elif dataset == "imagenet":
        with open(cfg_dataset.paths.save_path + "ImageNet_img_emb_clip.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + "ImageNet_text_emb_clip.pkl", "rb") as f:
            Data2 = pickle.load(f)
    elif dataset == "tiil":
        with open(cfg_dataset.paths.save_path + "TIIL_img_emb_clip.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + "TIIL_text_emb_clip.pkl", "rb") as f:
            Data2 = pickle.load(f)
    elif dataset == "cosmos":
        with open(cfg_dataset.paths.save_path + "COSMOS_img_emb_clip.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + "COSMOS_text_emb_clip.pkl", "rb") as f:
            Data2 = pickle.load(f)
    # TODO: add more datasets
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return cfg_dataset, Data1, Data2


def origin_centered(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """This function returns the origin centered data matrix and the mean of each feature.

    Args:
        X: data matrix (n_samples, n_features)

    Returns:
        origin centered data matrix, mean of each feature
    """
    return X - np.mean(X, axis=0), np.mean(X, axis=0)
