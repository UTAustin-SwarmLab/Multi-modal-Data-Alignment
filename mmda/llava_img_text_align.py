"""Script to query llava and ask if the img and text are aligned and save the aligned answer as pickle file."""

import pickle
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

import hydra
from mmda.utils.dataset_utils import (
    get_train_test_split_index,
    load_cosmos,
    load_imagenet,
    load_pitts,
    load_sop,
    load_tiil,
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
    llava_align(cfg)
    if cfg.dataset in cfg.dataset_level_datasets:
        llava_shuffle_align(cfg, "dataset")
    if cfg.dataset in cfg.class_level_datasets:
        llava_shuffle_align(cfg, "class")
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

    cfg_dataset = cfg[cfg.dataset]

    # load raw data
    if cfg.dataset == "sop":
        img_paths, text_descriptions, _, _ = load_sop(cfg_dataset)
    elif cfg.dataset == "imagenet":
        img_paths, mturks, orig_idx, clsidx_to_labels = load_imagenet(cfg_dataset)
        text_descriptions = []
        for i in range(len(orig_idx)):
            description = "An image of " + clsidx_to_labels[orig_idx[i]]
            text_descriptions.append(description)
    elif cfg.dataset == "tiil":
        img_paths, text_descriptions, _, _ = load_tiil(cfg_dataset)
    elif cfg.dataset == "cosmos":
        img_paths, text_descriptions, _, _ = load_cosmos(cfg_dataset)
    elif cfg.dataset == "pitts":
        img_paths, text_descriptions, _ = load_pitts(cfg_dataset)
    # TODO: add more datasets
    else:
        msg = f"Dataset {cfg.dataset} not implemented"
        raise NotImplementedError(msg)

    # split data
    if cfg.dataset in ("sop", "pitts"):
        train_idx, val_idx = get_train_test_split_index(
            cfg.train_test_ratio, len(img_paths)
        )
        _, img_paths = train_test_split(img_paths, train_idx, val_idx)
        _, text_descriptions = train_test_split(text_descriptions, train_idx, val_idx)
    elif cfg.dataset == "cosmos":  # cosmos has 2 captions (1ooc 1 in) per image
        img_paths = img_paths[-3400:][1::2]
        text_descriptions = text_descriptions[-3400:][1::2]

    # query llava without shuffling
    aligned_answer = llava_img_text_align(cfg, img_paths, text_descriptions)
    model_name = cfg.llava.model_path.split("/")[-1]

    Path(cfg_dataset.paths.save_path).mkdir(parents=True, exist_ok=True)
    # Save text_descriptions pickle
    with Path(
        cfg_dataset.paths.save_path + f"{cfg.dataset}_{model_name}_aligned.pkl"
    ).open("wb") as f:
        pickle.dump(aligned_answer, f)


def llava_shuffle_align(cfg: DictConfig, shuffle_level: str = "dataset") -> None:
    """Query llava and save the dataset level unaligned answer as pickle file.

    Args:
        cfg (DictConfig): config file
        shuffle_level (str): shuffle level. It can be "dataset", "class", or "object".

    Returns:
        None
    """
    # set random seed
    np.random.seed(cfg.seed)

    cfg_dataset = cfg[cfg.dataset]

    # load raw data
    if cfg.dataset == "sop":
        img_paths, text_descriptions, _, _ = load_sop(cfg_dataset)
    elif cfg.dataset == "pitts":
        img_paths, text_descriptions, _ = load_pitts(cfg_dataset)
    # split data
    train_idx, val_idx = get_train_test_split_index(
        cfg.train_test_ratio, len(img_paths)
    )
    train_img_path, val_img_path = train_test_split(img_paths, train_idx, val_idx)
    train_txt, val_txt = train_test_split(text_descriptions, train_idx, val_idx)

    train_txt_unalign, val_txt_unalign = shuffle_by_level(
        cfg, shuffle_level, train_txt, val_txt, train_idx, val_idx
    )
    model_name = cfg.llava.model_path.split("/")[-1]

    # query llava without shuffling
    aligned_answer = llava_img_text_align(cfg, val_img_path, val_txt_unalign)
    level_tag = (
        "ds"
        if shuffle_level == "dataset"
        else "class" if shuffle_level == "class" else "obj"
    )
    # Save text_descriptions pickle
    with Path(
        cfg_dataset.paths.save_path
        + f"{cfg.dataset}_{model_name}_{level_tag}_unalign.pkl"
    ).open("wb") as f:
        pickle.dump(aligned_answer, f)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 poetry run python mmda/llava_img_text_align.py
