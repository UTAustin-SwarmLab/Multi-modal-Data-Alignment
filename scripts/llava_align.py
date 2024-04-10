import pickle

import numpy as np
from omegaconf import DictConfig

import hydra
from mmda.data_real_align import parse_wrong_label
from mmda.utils.data_utils import (
    load_two_encoder_data,
)
from mmda.utils.dataset_utils import (
    get_train_test_split_index,
    train_test_split,
)


def parse_llava_yes_no(llava_output: list[str]) -> np.ndarray:
    """Parse the llava output to see if the answer is yes or no.

    Args:
        llava_output: lsit of llava output
    Returns:
        yes_no_array: numpy array of boolean values
    """
    llava_output_merge_threads = []
    for i in range(len(llava_output)):
        llava_output_merge_threads.extend(llava_output[i])
    yes_no_array = []
    for path_answer in llava_output_merge_threads:
        if "yes" in path_answer[1].lower():
            yes_no_array.append(True)
        else:
            yes_no_array.append(False)
    yes_no_array = np.array(yes_no_array).astype(bool)
    return yes_no_array


def boolean_binary_detection(align: np.ndarray, unalign: np.ndarray) -> list[list[float]]:
    """Calculate the ROC points for boolean data.

    Args:
        align: boolean outcome of aligned data
        unalign: boolean outcome of unaligned data
    Return
        ROC points
    """
    TP = np.sum(align)
    FP = np.sum(unalign)
    FN = np.sum(1 - align)
    TN = np.sum(1 - unalign)
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR


@hydra.main(version_base=None, config_path="../config", config_name="main")
def llava_align(cfg: DictConfig):  # noqa: D103
    # set random seed
    np.random.seed(cfg.seed)

    cfg_dataset, Data1, Data2 = load_two_encoder_data(cfg)

    # load image embeddings and text embeddings

    if cfg.dataset == "sop":
        with open(cfg_dataset.paths.save_path + f"{cfg.dataset}_llava-v1.5-13b_aligned.pkl", "rb") as f:
            align = pickle.load(f)
        with open(cfg_dataset.paths.save_path + f"{cfg.dataset}_llava-v1.5-13b_ds_unalign.pkl", "rb") as f:
            ds_unalign = pickle.load(f)
        with open(cfg_dataset.paths.save_path + f"{cfg.dataset}_llava-v1.5-13b_class_unalign.pkl", "rb") as f:
            class_unalign = pickle.load(f)
        with open(cfg_dataset.paths.save_path + f"{cfg.dataset}_llava-v1.5-13b_obj_unalign.pkl", "rb") as f:
            obj_unalign = pickle.load(f)

        align = parse_llava_yes_no(align)
        ds_unalign = parse_llava_yes_no(ds_unalign)
        class_unalign = parse_llava_yes_no(class_unalign)
        obj_unalign = parse_llava_yes_no(obj_unalign)

        # print ROC
        print("Aligned vs Unaligned")
        TPR, FPR = boolean_binary_detection(align, ds_unalign)
        print(f"TPR: {TPR}, FPR: {FPR}")
        print("Class Aligned vs Unaligned")
        TPR, FPR = boolean_binary_detection(align, class_unalign)
        print(f"TPR: {TPR}, FPR: {FPR}")
        print("Object Aligned vs Unaligned")
        TPR, FPR = boolean_binary_detection(align, obj_unalign)
        print(f"TPR: {TPR}, FPR: {FPR}")
        return
    # TODO: add more datasets
    elif cfg.dataset == "imagenet" or cfg.dataset == "tiil":
        with open(cfg_dataset.paths.save_path + f"{cfg.dataset}_llava-v1.5-13b_aligned.pkl", "rb") as f:
            llava_results = pickle.load(f)
    llava_results = parse_llava_yes_no(llava_results)
    wrong_labels_bool = parse_wrong_label(cfg)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Data1.shape[0])
    train_wrong_labels_bool, val_wrong_labels_bool = train_test_split(wrong_labels_bool, trainIdx, valIdx)
    train_llava_results, val_llava_results = train_test_split(llava_results, trainIdx, valIdx)

    # separate aligned data and unaligned data
    val_llava_resultsAlign = val_llava_results[~val_wrong_labels_bool]
    val_llava_resultsUnalign = val_llava_results[val_wrong_labels_bool]

    # print ROC
    print("Aligned vs Unaligned")
    TPR, FPR = boolean_binary_detection(val_llava_resultsAlign, val_llava_resultsUnalign)
    print(f"TPR: {TPR}, FPR: {FPR}")
    return


if __name__ == "__main__":
    llava_align()
