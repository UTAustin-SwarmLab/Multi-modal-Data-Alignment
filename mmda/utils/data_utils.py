import os
import pickle

import datasets
import numpy as np
import pandas as pd
from omegaconf import DictConfig


def load_dataset_config(cfg: DictConfig) -> DictConfig:
    """Load the configuration file for the dataset.

    Args:
        cfg: configuration file
    Returns:
        cfg_dataset: configuration file for the dataset
    """
    dataset = cfg.dataset
    if dataset == "sop":
        cfg_dataset = cfg.sop
    elif dataset == "musiccaps":
        cfg_dataset = cfg.musiccaps
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return cfg_dataset


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
    if dataset == "sop":
        # load image embeddings and text embeddings
        with open(cfg_dataset.paths.save_path + f"data/SOP_img_emb_{cfg_dataset.img_encoder}.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + f"data/SOP_text_emb_{cfg_dataset.text_encoder}.pkl", "rb") as f:
            Data2 = pickle.load(f)
    elif dataset == "musiccaps":
        # load image embeddings and text embeddings
        with open(cfg_dataset.paths.save_path + f"MusicCaps_audio_emb_{cfg_dataset.audio_encoder}.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + f"MusicCaps_text_emb_{cfg_dataset.text_encoder}.pkl", "rb") as f:
            Data2 = pickle.load(f)
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
    if dataset == "sop":
        # load image embeddings and text embeddings
        with open(cfg_dataset.paths.save_path + "data/SOP_img_emb_clip.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + "data/SOP_text_emb_clip.pkl", "rb") as f:
            Data2 = pickle.load(f)
    elif dataset == "musiccaps":
        # load image embeddings and text embeddings
        with open(cfg_dataset.paths.save_path + "MusicCaps_audio_emb_clap.pkl", "rb") as f:
            Data1 = pickle.load(f)
        with open(cfg_dataset.paths.save_path + "MusicCaps_text_emb_clap.pkl", "rb") as f:
            Data2 = pickle.load(f)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return cfg_dataset, Data1, Data2


def load_MusicCaps(cfg_dataset: DictConfig) -> tuple[list[str], list[str]]:
    """Load the Google MusicCaps dataset.

    Args:
        cfg_dataset: configuration file
    Returns:
        audio paths, text descriptions, aspect_list, and audioset_positive_labels
    """
    parent_dir = os.path.abspath(os.path.join(cfg_dataset.paths.dataset_path, os.pardir))
    df_path = os.path.join(parent_dir, "MusicCaps_parsed.csv")
    if os.path.exists(df_path):
        dataframe = pd.read_csv(df_path)
        return dataframe

    # if the parsed file does not exist, load the dataset and parse it
    dataset = datasets.load_dataset("google/MusicCaps", split="train")
    dataframe = pd.DataFrame(
        columns=[
            "ytid",
            "audio_path",
            "caption",
            "aspect_list",
            "audioset_positive_labels",
            "start_time",
            "end_time",
        ]
    )
    rows_list = []
    for data in dataset:
        audio_path = os.path.join(cfg_dataset.paths.dataset_path, f"{data['ytid']}.wav")
        ### check if the audio file exists
        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} does not exist. Skipping.")
            continue

        row_in_dict = {
            "ytid": data["ytid"],
            "audio_path": audio_path,
            "caption": data["caption"],
            "aspect_list": data["aspect_list"],
            "audioset_positive_labels": data["audioset_positive_labels"][0],
            "start_s": data["start_s"],
            "end_s": data["end_s"],
        }
        rows_list.append(row_in_dict)

    dataframe = pd.DataFrame(rows_list)
    dataframe.to_csv(df_path, index=False)
    return dataframe


def load_SOP(cfg_dataset: DictConfig) -> tuple[list[str], list[str]]:
    """Load the Stanford Online Products dataset.

    Args:
        cfg_dataset: configuration file
    Returns:
        image paths, text descriptions, classes, and object ids
    """
    # load SOP images path
    with open(cfg_dataset.paths.dataset_path + "text_descriptions_SOP.pkl", "rb") as f:
        # '/store/omama/datasets/Stanford_Online_Products/bicycle_final/251952414262_2.JPG'
        # "The image features a close-up view of a bicycle's suspension system,
        # specifically focusing on the front fork and the shock absorber.</s>"
        path_text_descriptions = pickle.load(f)
    for path_text in path_text_descriptions:
        path_text[0] = path_text[0].replace("/store/", "/nas/")
        path_text[1] = path_text[1].replace("</s>", "")
    img_paths = [x[0] for x in path_text_descriptions]
    text_descriptions = [x[1] for x in path_text_descriptions]
    ### img_path example: /store/omama/datasets/Stanford_Online_Products/bicycle_final/251952414262_2.JPG
    classes = [img_path.split("/")[-2].split("_")[0] for img_path in img_paths]
    obj_ids = [img_path.split("/")[-1].split("_")[0] for img_path in img_paths]
    return img_paths, text_descriptions, classes, obj_ids


def origin_centered(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """This function returns the origin centered data matrix and the mean of each feature.

    Args:
        X: data matrix (n_samples, n_features)

    Returns:
        origin centered data matrix, mean of each feature
    """
    return X - np.mean(X, axis=0), np.mean(X, axis=0)


def get_train_test_split_index(train_test_ration: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the index of the training and validation set.

    Args:
        train_test_ration: ratio of training set
        N: number of samples
    Returns:
        index of the training and validation set
    """
    arange = np.arange(N)
    np.random.shuffle(arange)
    trainIdx = arange[: int(N * train_test_ration)]
    valIdx = arange[int(N * train_test_ration) :]
    return trainIdx, valIdx


def train_test_split(data: np.ndarray, train_idx: list[int], val_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Split the data into training and validation set.

    Args:
        data: data
        train_idx: index of the training set
        val_idx: index of the validation set
    Return:
        training and validation set
    """
    if type(data) != np.ndarray:
        data = np.array(data)
    return data[train_idx], data[val_idx]


def filter_str_label(ground_truth: np.ndarray) -> dict[str, np.ndarray]:
    """Filter the data based on the provided ground truth.

    Args:
        ground_truth: ground truth. shape: (N, )

    Return:
        a dict of index filter. keys: unique ground truth, values: indices of the data
    """
    ground_truth = ground_truth.astype(str)
    unique_classes = np.unique(ground_truth)
    filter_idx = {}
    for cls in unique_classes:
        filter_idx[cls] = np.where(ground_truth == cls)[0]
    return filter_idx


def shuffle_data_by_indices(data: np.ndarray, filter_idx: dict[str, np.ndarray]) -> np.ndarray:
    """Shuffle the data by classes.

    Args:
        data: data
        filter_idx: a dict of index filter. keys: unique ground truth, values: indices of the data
        seed: random seed
    Return:
        shuffled data
    """
    for key, val in filter_idx.items():
        c = data[val]
        np.random.shuffle(c)
        data[val] = c
    return data


def filter_outliers(scores: np.ndarray, threshold: float, right_tail: bool = False) -> np.ndarray:
    """Return the indices of the outliers (either the right or left tails) filtered from the given the scores.

    Args:
        scores: scores of data. shape: (N, ).
        threshold: threshold of similarity score for outliers.
        right_tail: right tail or left tail of the data.

    Return:
        indices of the outliers.
    """
    if right_tail:
        index = np.where(scores > threshold)[0]
        # sort the index from high to low by the similarity score
        index = index[np.argsort(scores[index])[::-1]]  # descending order
    else:
        index = np.where(scores < threshold)[0]
        # sort the index from low to high by the similarity score
        index = index[np.argsort(scores[index])]  # ascending order
    return index
