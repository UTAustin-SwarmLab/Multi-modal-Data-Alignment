import pickle
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig

import hydra


@hydra.main(version_base=None, config_path='config', config_name='sop')
def load_SOP(cfg: DictConfig) -> Tuple[List[str], List[str]]:
    """
    Load the Stanford Online Products dataset
    :param cfg: configuration file
    :return: image paths and text descriptions
    """
    # load SOP images path
    with open(cfg.paths.sop_dataset_path + "text_descriptions_SOP.pkl", 'rb') as f:
        # '/store/omama/datasets/Stanford_Online_Products/bicycle_final/251952414262_2.JPG'
        # "The image features a close-up view of a bicycle's suspension system, specifically focusing on the front fork and the shock absorber.</s>"
        path_text_descriptions = pickle.load(f) 
    for path_text in path_text_descriptions:
        path_text[0] = path_text[0].replace('/store/', '/nas/')
        path_text[1] = path_text[1].replace('</s>', '')
    img_paths = [x[0] for x in path_text_descriptions]
    text_descriptions = [x[1] for x in path_text_descriptions]

    ### img_path example: /store/omama/datasets/Stanford_Online_Products/bicycle_final/251952414262_2.JPG
    classes = [img_path.split('/')[-2].split('_')[0] for img_path in img_paths]
    obj_ids = [img_path.split('/')[-1].split('_')[0] for img_path in img_paths]
    return img_paths, text_descriptions, classes, obj_ids

def origin_centered(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ''' This function returns the origin centered data matrix and the mean of each feature
    Args:
        X: data matrix (n_samples, n_features)
    Returns: 
        origin centered data matrix, mean of each feature
    '''
    return X - np.mean(X, axis=0), np.mean(X, axis=0)

def get_train_test_split_index(train_test_ration: float, N: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the index of the training and validation set
    :param train_test_ration: ratio of training set
    :param N: number of samples
    :param seed: random seed
    :return: index of the training and validation set
    """
    np.random.seed(seed)
    arange = np.arange(N)
    np.random.shuffle(arange)
    trainIdx = arange[:int(N * train_test_ration)]
    valIdx = arange[int(N * train_test_ration):]

    return trainIdx, valIdx

def train_test_split(data: np.ndarray, train_idx: List[int], val_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the data into training and validation set
    :param data: data
    :param train_idx: index of the training set
    :param val_idx: index of the validation set
    :return: training and validation set
    """
    if type(data) != np.ndarray:
        # print("Converting data to numpy array")
        data = np.array(data)
    return data[train_idx], data[val_idx]

def filter_str_label(ground_truth: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Filter the data based on the provided ground truth
    :param ground_truth: ground truth. shape: (N, )
    :return: a dict of index filter. keys: unique ground truth, values: indices of the data
    """
    ground_truth = ground_truth.astype(str)
    unique_classes = np.unique(ground_truth)
    filter_idx = {}
    for cls in unique_classes:
        filter_idx[cls] = np.where(ground_truth == cls)[0]
    return filter_idx

def shuffle_data_by_indices(data: np.ndarray, filter_idx: Dict[str, np.ndarray], seed: int =42) -> np.ndarray:
    """
    Shuffle the data by classes
    :param data: data
    :param classes: classes
    :param filter_idx: a dict of index filter. keys: unique ground truth, values: indices of the data
    :return: shuffled data
    """
    np.random.seed(seed)
    for key, val in filter_idx.items():
        # print(f"{key}: {len(val)}")
        c = data[val]
        np.random.shuffle(c)
        data[val] = c
    return data