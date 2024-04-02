import os
import pickle

import datasets
import numpy as np
import pandas as pd
from omegaconf import DictConfig


def load_MusicCaps(cfg: DictConfig) -> tuple[list[str], list[str]]:
    """Load the Google MusicCaps dataset.

    Args:
        cfg: configuration file
    Returns:
        audio paths, text descriptions, aspect_list, and audioset_positive_labels
    """
    dataset = datasets.load_dataset("google/MusicCaps", split="train")
    dataframe = pd.DataFrame(
        columns=[
            "ytid",
            "audio_path",
            "text_description",
            "aspect_list",
            "audioset_positive_labels",
            "start_time",
            "end_time",
        ]
    )
    for data in dataset:
        audio_path = os.path.join(cfg.paths.dataset_path, f"{data['ytid']}.wav")
        ### check if the audio file exists
        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} does not exist. Skipping.")
            continue
        else:
            dataframe = dataframe.append(
                {
                    "ytid": data["ytid"],
                    "audio_path": audio_path,
                    "text_description": data["text_description"],
                    "aspect_list": data["aspect_list"],
                    "audioset_positive_labels": data["audioset_positive_labels"],
                    "start_time": data["start_time"],
                    "end_time": data["end_time"],
                },
                ignore_index=True,
            )

    return


def load_SOP(cfg: DictConfig) -> tuple[list[str], list[str]]:
    """Load the Stanford Online Products dataset.

    Args:
        cfg: configuration file
    Returns:
        image paths, text descriptions, classes, and object ids
    """
    # load SOP images path
    with open(cfg.paths.dataset_path + "text_descriptions_SOP.pkl", "rb") as f:
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


def get_train_test_split_index(train_test_ration: float, N: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the index of the training and validation set.

    Args:
        train_test_ration: ratio of training set
        N: number of samples
        seed: random seed
    Returns:
        index of the training and validation set
    """
    np.random.seed(seed)
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
        # print("Converting data to numpy array")
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


def shuffle_data_by_indices(data: np.ndarray, filter_idx: dict[str, np.ndarray], seed: int = 42) -> np.ndarray:
    """Shuffle the data by classes.

    Args:
        data: data
        filter_idx: a dict of index filter. keys: unique ground truth, values: indices of the data
        seed: random seed
    Return:
        shuffled data
    """
    np.random.seed(seed)
    for key, val in filter_idx.items():
        # print(f"{key}: {len(val)}")
        c = data[val]
        np.random.shuffle(c)
        data[val] = c
    return data


def filter_outliers(sim_scores: np.ndarray, threshold: float, right_tail: bool = True) -> np.ndarray:
    """Show the outliers.

    Args:
        sim_scores: similarity scores. shape: (N, ).
        threshold: threshold of similarity score for outliers.
        right_tail: right tail or left tail.

    Return:
        index of the outliers.
    """
    index = np.where(sim_scores > threshold)[0] if right_tail else np.where(sim_scores < threshold)[0]
    print(f"Number of outliers: {len(index)}")
    ### sort the index from high to low by the similarity score
    index = index[np.argsort(sim_scores[index])[::-1]]
    return index
