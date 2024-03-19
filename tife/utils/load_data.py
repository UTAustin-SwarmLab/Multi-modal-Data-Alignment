import pickle
from typing import List, Tuple

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
    with open(cfg.sop_dataset_path + "text_descriptions_SOP.pkl", 'rb') as f:
        # '/store/omama/datasets/Stanford_Online_Products/bicycle_final/251952414262_2.JPG'
        # "The image features a close-up view of a bicycle's suspension system, specifically focusing on the front fork and the shock absorber.</s>"
        path_text_descriptions = pickle.load(f) 
    for path_text in path_text_descriptions:
        path_text[0] = path_text[0].replace('/store/', '/nas/')
        path_text[1] = path_text[1].replace('</s>', '')
    img_paths = [x[0] for x in path_text_descriptions]
    text_descriptions = [x[1] for x in path_text_descriptions]

    ### example: /store/omama/datasets/Stanford_Online_Products/bicycle_final/251952414262_2.JPG
    classes = [img_path.split('/')[-2].split('_')[0] for img_path in img_paths]
    obj_ids = [img_path.split('/')[-1].split('_')[0] for img_path in img_paths]
    return img_paths, text_descriptions, classes, obj_ids

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
    return data[train_idx], data[val_idx]