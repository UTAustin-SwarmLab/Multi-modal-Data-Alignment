import json
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
    elif dataset == "imagenet":
        cfg_dataset = cfg.imagenet
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return cfg_dataset


def load_ImageNet(cfg_dataset: DictConfig) -> tuple[list[str], list[int], list[int], dict[int, str]]:
    """Load the ImageNet dataset.

    Args:
        cfg_dataset: configuration file
    Returns:
        image paths, MTurk verified classe indices, ground truth class indices, and dict of class idx to str.
    """
    # load json file
    with open(os.path.join(cfg_dataset.paths.dataset_path, "imagenet_mturk.json")) as f:
        mturks = json.load(f)  # 5440
        """
        {
            "id": 293,
            "url": "https://labelerrors.com//static/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG",
            "given_original_label": 0,
            "given_original_label_name": "tench",
            "our_guessed_label": 48,
            "our_guessed_label_name": "Komodo dragon",
            "mturk": {
            "given": 5,
            "guessed": 0,
            "neither": 0,
            "both": 0
            }
        },
        """
    with open(os.path.join(cfg_dataset.paths.dataset_path, "imagenet_val_set_index_to_filepath.json")) as f:
        idx2path = json.load(f)  # ["val/n01440764/ILSVRC2012_val_00000293.JPEG", ...] # 50000
    img_path = [None] * len(idx2path)
    for idx, path in enumerate(idx2path):
        img_path[idx] = os.path.join(cfg_dataset.paths.dataset_path, path)
    orig_idx = np.load(os.path.join(cfg_dataset.paths.dataset_path, "imagenet_val_set_original_labels.npy"))

    mturks_idx = orig_idx.copy()
    # correct the labels to the MTurk labels
    for mturk in mturks:
        orig_label = mturk["given_original_label"]
        guessed_label = mturk["our_guessed_label"]
        img_name = mturk["url"].replace("https://labelerrors.com//static/imagenet/", "")
        img_index = idx2path.index(img_name)
        mturks_idx[img_index] = int(guessed_label)
        assert orig_idx[img_index] == orig_label, f"Mismatch at {img_index}: {orig_idx[img_index]} != {orig_label}"
    assert np.sum(orig_idx != mturks_idx) == len(mturks), f"Relabel num mismatch: {np.sum(orig_idx != mturks_idx)}"

    # convert the labels to string. Obtained from https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    with open(os.path.join(cfg_dataset.paths.dataset_path, "ImageNet_clsidx_to_labels.txt")) as f:
        clsidx_to_labels_txt = f.readlines()
    clsidx_to_labels = {}
    for line in clsidx_to_labels_txt:  # example: {0: 'tench, Tinca tinca',
        line = line.replace("{", "").replace("}", "").rstrip(",\n")
        idx, label = line.split(":")
        idx, label = int(idx.strip()), label.strip()
        label = label.replace("'", "")
        clsidx_to_labels[idx] = label
    return img_path, mturks_idx, orig_idx, clsidx_to_labels


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


def shuffle_by_level(
    cfg_dataset: DictConfig,
    dataset: str,
    shuffle_level: str,
    trainData2Unalign: np.ndarray,
    valData2Unalign: np.ndarray,
    trainIdx: list[int],
    valIdx: list[int],
):
    """Shuffle the data by dataset, class, or object level.

    Args:
        cfg_dataset: configuration file
        dataset: dataset name
        shuffle_level: shuffle level. It can only be "dataset", "class", or "object".
        trainData2Unalign: unaligned data
        valData2Unalign: unaligned data
        trainIdx: training indices
        valIdx: validation indices
    Returns:
        shuffled Data2
    """
    assert shuffle_level in ["dataset", "class", "object"], f"shuffle_level {shuffle_level} not supported."
    # sop shuffle by class or object
    if shuffle_level == "dataset":
        np.random.shuffle(trainData2Unalign)
        np.random.shuffle(valData2Unalign)
        return trainData2Unalign, valData2Unalign
    if dataset == "sop":
        _, _, classes, obj_ids = load_SOP(cfg_dataset)
        if shuffle_level == "class":
            train_gts, val_gts = train_test_split(classes, trainIdx, valIdx)
        elif shuffle_level == "object":
            train_gts, val_gts = train_test_split(obj_ids, trainIdx, valIdx)
        else:
            raise ValueError(f"Dataset {dataset} does not have {shuffle_level} information.")
    elif dataset == "musiccaps":
        dataframe = load_MusicCaps(cfg_dataset)
        if shuffle_level == "class":
            gts = dataframe["audioset_positive_labels"].tolist()
            train_gts, val_gts = train_test_split(gts, trainIdx, valIdx)
        else:
            raise ValueError(f"Dataset {dataset} does not have {shuffle_level} information.")
    elif dataset == "imagenet":
        _, _, orig_idx, clsidx_to_labels = load_ImageNet(cfg_dataset)
        orig_labels = [clsidx_to_labels[i] for i in orig_idx]
        if shuffle_level == "class":
            train_gts, val_gts = train_test_split(orig_labels, trainIdx, valIdx)
        else:
            raise ValueError(f"Dataset {dataset} does not have {shuffle_level} information.")
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    val_dict_filter = filter_str_label(val_gts)
    valData2Unalign = shuffle_data_by_indices(valData2Unalign, val_dict_filter)
    train_dict_filter = filter_str_label(train_gts)
    trainData2Unalign = shuffle_data_by_indices(trainData2Unalign, train_dict_filter)
    return trainData2Unalign, valData2Unalign


# import hydra
# @hydra.main(version_base=None, config_path="../../config", config_name="main")
# def test(cfg: DictConfig):
#     idx2path, mturks_idx, orig_idx, clsidx_to_labels = load_ImageNet(cfg.imagenet)
# if __name__ == "__main__":
#     test()
