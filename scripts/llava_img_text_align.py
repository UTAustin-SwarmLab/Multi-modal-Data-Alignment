import os
import pickle

import numpy as np
from omegaconf import DictConfig

import hydra
from mmda.utils.data_utils import (
    load_dataset_config,
)
from mmda.utils.dataset_utils import (
    filter_str_label,
    get_train_test_split_index,
    load_COSMOS,
    load_ImageNet,
    load_PITTS,
    load_SOP,
    load_TIIL,
    shuffle_data_by_indices,
    train_test_split,
)
from mmda.utils.llava_utils import llava_img_text_align


@hydra.main(version_base=None, config_path="../config", config_name="main")
def llava_align(cfg: DictConfig) -> None:
    """Query llava and save the aligned answer as pickle file.

    Args:
        cfg (DictConfig): config file

    Returns:
        None
    """
    # set random seed
    np.random.seed(cfg.seed)

    cfg_dataset = load_dataset_config(cfg)

    # load raw data
    if cfg.dataset == "sop":
        img_paths, text_descriptions, _, _ = load_SOP(cfg_dataset)
    elif cfg.dataset == "imagenet":
        img_paths, Mturks, orig_idx, clsidx_to_labels = load_ImageNet(cfg_dataset)
        text_descriptions = []
        for i in range(len(orig_idx)):
            description = "An image of " + clsidx_to_labels[orig_idx[i]]
            text_descriptions.append(description)
    elif cfg.dataset == "tiil":
        img_paths, text_descriptions, _, _ = load_TIIL(cfg_dataset)
    elif cfg.dataset == "cosmos":
        img_paths, text_descriptions, _, _ = load_COSMOS(cfg_dataset)
    elif cfg.dataset == "pitts":
        img_paths, text_descriptions, _ = load_PITTS(cfg_dataset)
    # TODO: add more datasets
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} not implemented")

    # split data
    if cfg.dataset == "sop":
        trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, len(img_paths))
        _, img_paths = train_test_split(img_paths, trainIdx, valIdx)
        _, text_descriptions = train_test_split(text_descriptions, trainIdx, valIdx)

    # query llava without shuffling
    aligned_answer = llava_img_text_align(cfg, img_paths, text_descriptions)
    model_name = cfg.llava.model_path.split("/")[-1]

    os.makedirs(cfg_dataset.paths.save_path, exist_ok=True)
    # Save text_descriptions pickle
    with open(
        cfg_dataset.paths.save_path + f"{cfg.dataset}_{model_name}_aligned.pkl",
        "wb",
    ) as f:
        pickle.dump(aligned_answer, f)

    return


@hydra.main(version_base=None, config_path="../config", config_name="main")
def llava_dataset_shuffle(cfg: DictConfig):
    """Query llava and save the dataset level unaligned answer as pickle file.

    Args:
        cfg (DictConfig): config file

    Returns:
        None
    """
    # set random seed
    np.random.seed(cfg.seed)

    cfg_dataset = load_dataset_config(cfg)

    # load raw data
    if cfg.dataset == "sop":
        img_paths, text_descriptions, _, _ = load_SOP(cfg_dataset)
    elif cfg.dataset == "pitts":
        img_paths, text_descriptions, _ = load_PITTS(cfg_dataset)

    # split data
    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, len(img_paths))
    _, valImgPath = train_test_split(img_paths, trainIdx, valIdx)
    trainTxt, valTxt = train_test_split(text_descriptions, trainIdx, valIdx)
    np.random.shuffle(trainTxt)
    np.random.shuffle(valTxt)

    model_name = cfg.llava.model_path.split("/")[-1]

    # query llava without shuffling
    aligned_answer = llava_img_text_align(cfg, valImgPath, valTxt)
    # Save text_descriptions pickle
    with open(
        cfg_dataset.paths.save_path + f"{cfg.dataset}_{model_name}_ds_unalign.pkl",
        "wb",
    ) as f:
        pickle.dump(aligned_answer, f)

    return


@hydra.main(version_base=None, config_path="../config", config_name="main")
def llava_class_shuffle(cfg: DictConfig):
    """Query llava and save the class level unaligned answer as pickle file.

    Args:
        cfg (DictConfig): config file

    Returns:
        None
    """
    # set random seed
    np.random.seed(cfg.seed)

    cfg_dataset = load_dataset_config(cfg)

    # load raw data
    if cfg.dataset == "sop":
        img_paths, text_descriptions, classes, _ = load_SOP(cfg_dataset)
    elif cfg.dataset == "pitts":
        img_paths, text_descriptions, _ = load_PITTS(cfg_dataset)

    # split data
    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, len(img_paths))
    _, valImgPath = train_test_split(img_paths, trainIdx, valIdx)
    _, valTxt = train_test_split(text_descriptions, trainIdx, valIdx)
    _, valClasses = train_test_split(classes, trainIdx, valIdx)

    # filter and shuffle data by classes or object ids
    val_class_dict_filter = filter_str_label(valClasses)
    valTxt = shuffle_data_by_indices(valTxt, val_class_dict_filter, seed=cfg.seed)

    model_name = cfg.llava.model_path.split("/")[-1]

    # query llava without shuffling
    class_unalign_answer = llava_img_text_align(cfg, valImgPath, valTxt)
    # Save text_descriptions pickle
    with open(
        cfg_dataset.paths.save_path + f"{cfg.dataset}_{model_name}_class_unalign.pkl",
        "wb",
    ) as f:
        pickle.dump(class_unalign_answer, f)

    return


@hydra.main(version_base=None, config_path="../config", config_name="main")
def llava_obj_shuffle(cfg: DictConfig):
    """Query llava and save the object level unaligned answer as pickle file.

    Args:
        cfg (DictConfig): config file

    Returns:
        None
    """
    # set random seed
    np.random.seed(cfg.seed)

    cfg_dataset = load_dataset_config(cfg)

    # load raw data
    if cfg.dataset == "sop":
        img_paths, text_descriptions, _, obj_ids = load_SOP(cfg_dataset)
    # split data
    elif cfg.dataset == "pitts":
        img_paths, text_descriptions, _ = load_PITTS(cfg_dataset)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, len(img_paths))
    _, valImgPath = train_test_split(img_paths, trainIdx, valIdx)
    _, valTxt = train_test_split(text_descriptions, trainIdx, valIdx)
    _, valObjIds = train_test_split(obj_ids, trainIdx, valIdx)

    # filter and shuffle data by classes or object ids
    val_obj_dict_filter = filter_str_label(valObjIds)
    valTxt = shuffle_data_by_indices(valTxt, val_obj_dict_filter, seed=cfg.seed)

    model_name = cfg.llava.model_path.split("/")[-1]

    # query llava without shuffling
    obj_unalign_answer = llava_img_text_align(cfg, valImgPath, valTxt)
    # Save text_descriptions pickle
    with open(
        cfg_dataset.paths.save_path + f"{cfg.dataset}_{model_name}_obj_unalign.pkl",
        "wb",
    ) as f:
        pickle.dump(obj_unalign_answer, f)

    return


if __name__ == "__main__":
    llava_align()
    llava_dataset_shuffle()
    llava_class_shuffle()
    llava_obj_shuffle()

# CUDA_VISIBLE_DEVICES=2 poetry run scripts/python query_llava.py
