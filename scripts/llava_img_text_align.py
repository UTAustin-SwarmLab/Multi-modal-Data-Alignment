import os
import pickle

import numpy as np
from omegaconf import DictConfig

import hydra
from mmda.utils.data_utils import (
    load_dataset_config,
)
from mmda.utils.dataset_utils import (
    get_train_test_split_index,
    load_COSMOS,
    load_ImageNet,
    load_PITTS,
    load_SOP,
    load_TIIL,
    shuffle_by_level,
    train_test_split,
)
from mmda.utils.llava_utils import llava_img_text_align


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main function to query llava and save the aligned answer as pickle file.

    Args:
        cfg (DictConfig): config file

    Returns:
        None
    """
    # llava_align(cfg)
    # if cfg.dataset in cfg.dataset_level_datasets:
    #     llava_shuffle_align(cfg, "dataset")
    # if cfg.dataset in cfg.class_level_datasets:
    #     llava_shuffle_align(cfg, "class")
    if cfg.dataset in cfg.object_level_datasets:
        llava_shuffle_align(cfg, "object")


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
    if cfg.dataset == "sop" or cfg.dataset == "pitts":
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


def llava_shuffle_align(cfg: DictConfig, shuffle_level: str = "dataset"):
    """Query llava and save the dataset level unaligned answer as pickle file.

    Args:
        cfg (DictConfig): config file
        shuffle_level (str): shuffle level. It can be "dataset", "class", or "object".

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
    trainImgPath, valImgPath = train_test_split(img_paths, trainIdx, valIdx)
    trainTxt, valTxt = train_test_split(text_descriptions, trainIdx, valIdx)

    trainTxtUnalign, valTxtUnalign = shuffle_by_level(
        cfg_dataset, cfg.dataset, shuffle_level, trainTxt, valTxt, trainIdx, valIdx
    )
    model_name = cfg.llava.model_path.split("/")[-1]

    # query llava without shuffling
    aligned_answer = llava_img_text_align(cfg, valImgPath, valTxtUnalign)
    level_tag = "ds" if shuffle_level == "dataset" else "class" if shuffle_level == "class" else "obj"
    # Save text_descriptions pickle
    with open(
        cfg_dataset.paths.save_path + f"{cfg.dataset}_{model_name}_{level_tag}_unalign.pkl",
        "wb",
    ) as f:
        pickle.dump(aligned_answer, f)
    return


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 poetry run python scripts/llava_img_text_align.py
