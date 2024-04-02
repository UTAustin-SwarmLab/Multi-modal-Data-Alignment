import pickle

import numpy as np
from omegaconf import DictConfig

from mmda.utils.data_utils import (
    filter_str_label,
    get_train_test_split_index,
    load_SOP,
    shuffle_data_by_indices,
    train_test_split,
)
from mmda.utils.hydra_utils import hydra_main
from mmda.utils.query_llava import query_llava


@hydra_main(version_base=None, config_path="config", config_name="sop")
def sop_llava_align(cfg: DictConfig) -> None:
    """Query llava and save the aligned answer as pickle file.

    Args:
        cfg (DictConfig): config file

    Returns:
        None
    """
    # set random seed
    np.random.seed(cfg.seed)
    # load raw data
    img_paths, text_descriptions, _, _ = load_SOP(cfg)
    # split data
    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, len(img_paths), cfg.seed)
    _, valImgPath = train_test_split(img_paths, trainIdx, valIdx)
    _, valTxt = train_test_split(text_descriptions, trainIdx, valIdx)

    model_name = cfg.llava.model_path.split("/")[-1]

    # query llava without shuffling
    aligned_answer = query_llava(cfg, valImgPath, valTxt)
    # Save text_descriptions pickle
    with open(
        cfg.paths.save_path + f"sop_{model_name}_aligned.pkl",
        "wb",
    ) as f:
        pickle.dump(aligned_answer, f)

    return


@hydra_main(version_base=None, config_path="config", config_name="sop")
def sop_llava_dataset_shuffle(cfg: DictConfig):
    """Query llava and save the dataset level unaligned answer as pickle file.

    Args:
        cfg (DictConfig): config file

    Returns:
        None
    """
    # set random seed
    np.random.seed(cfg.seed)
    # load raw data
    img_paths, text_descriptions, _, _ = load_SOP(cfg)
    # split data
    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, len(img_paths), cfg.seed)
    _, valImgPath = train_test_split(img_paths, trainIdx, valIdx)
    trainTxt, valTxt = train_test_split(text_descriptions, trainIdx, valIdx)
    np.random.shuffle(trainTxt)
    np.random.shuffle(valTxt)

    model_name = cfg.llava.model_path.split("/")[-1]

    # query llava without shuffling
    aligned_answer = query_llava(cfg, valImgPath, valTxt)
    # Save text_descriptions pickle
    with open(
        cfg.paths.save_path + f"sop_{model_name}_ds_unalign.pkl",
        "wb",
    ) as f:
        pickle.dump(aligned_answer, f)

    return


@hydra_main(version_base=None, config_path="config", config_name="sop")
def sop_llava_class_shuffle(cfg: DictConfig):
    """Query llava and save the class level unaligned answer as pickle file.

    Args:
        cfg (DictConfig): config file

    Returns:
        None
    """
    # set random seed
    np.random.seed(cfg.seed)
    # load raw data
    img_paths, text_descriptions, classes, _ = load_SOP(cfg)
    # split data
    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, len(img_paths), cfg.seed)
    _, valImgPath = train_test_split(img_paths, trainIdx, valIdx)
    _, valTxt = train_test_split(text_descriptions, trainIdx, valIdx)
    _, valClasses = train_test_split(classes, trainIdx, valIdx)

    # filter and shuffle data by classes or object ids
    val_class_dict_filter = filter_str_label(valClasses)
    valTxt = shuffle_data_by_indices(valTxt, val_class_dict_filter, seed=cfg.seed)

    model_name = cfg.llava.model_path.split("/")[-1]

    # query llava without shuffling
    class_unalign_answer = query_llava(cfg, valImgPath, valTxt)
    # Save text_descriptions pickle
    with open(
        cfg.paths.save_path + f"sop_{model_name}_class_unalign.pkl",
        "wb",
    ) as f:
        pickle.dump(class_unalign_answer, f)

    return


@hydra_main(version_base=None, config_path="config", config_name="sop")
def sop_llava_obj_shuffle(cfg: DictConfig):
    """Query llava and save the object level unaligned answer as pickle file.

    Args:
        cfg (DictConfig): config file

    Returns:
        None
    """
    # set random seed
    np.random.seed(cfg.seed)
    # load raw data
    img_paths, text_descriptions, _, obj_ids = load_SOP(cfg)
    # split data
    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, len(img_paths), cfg.seed)
    _, valImgPath = train_test_split(img_paths, trainIdx, valIdx)
    _, valTxt = train_test_split(text_descriptions, trainIdx, valIdx)
    _, valObjIds = train_test_split(obj_ids, trainIdx, valIdx)

    # filter and shuffle data by classes or object ids
    val_obj_dict_filter = filter_str_label(valObjIds)
    valTxt = shuffle_data_by_indices(valTxt, val_obj_dict_filter, seed=cfg.seed)

    model_name = cfg.llava.model_path.split("/")[-1]

    # query llava without shuffling
    obj_unalign_answer = query_llava(cfg, valImgPath, valTxt)
    # Save text_descriptions pickle
    with open(
        cfg.paths.save_path + f"sop_{model_name}_obj_unalign.pkl",
        "wb",
    ) as f:
        pickle.dump(obj_unalign_answer, f)

    return


if __name__ == "__main__":
    sop_llava_align()
    sop_llava_dataset_shuffle()
    sop_llava_class_shuffle()
    sop_llava_obj_shuffle()

# CUDA_VISIBLE_DEVICES=2 poetry run python sop_query_llava.py
