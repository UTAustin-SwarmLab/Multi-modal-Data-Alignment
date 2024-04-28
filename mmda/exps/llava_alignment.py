"""Parse the llava output for detecting mislabeled data or shuffled text descriptions and calculate the ROC curve."""

from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig

import hydra
from mmda.exps.mislabel_align import parse_wrong_label
from mmda.utils.data_utils import load_two_encoder_data
from mmda.utils.dataset_utils import get_train_test_split_index, train_test_split


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
    return np.array(yes_no_array).astype(bool)


def boolean_binary_detection(
    align: np.ndarray, unalign: np.ndarray
) -> list[list[float]]:
    """Calculate the ROC points for boolean data.

    Args:
        align: boolean outcome of aligned data
        unalign: boolean outcome of unaligned data
    Return
        ROC points
    """
    tp = np.sum(align)
    fp = np.sum(unalign)
    fn = np.sum(1 - align)
    tn = np.sum(1 - unalign)
    print(f"llava tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}")
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr


def llava_shuffle_align(
    cfg: DictConfig, shuffle_level: str = "dataset"
) -> list[list[float]]:
    """Return llava's shuffled alignment answer."""
    # set random seed
    cfg_dataset = cfg[cfg.dataset]

    # load image embeddings and text embeddings
    align = joblib.load(
        Path(cfg_dataset.paths.save_path + f"{cfg.dataset}_llava-v1.5-13b_aligned.pkl")
    )
    align = parse_llava_yes_no(align)
    if shuffle_level == "dataset":
        ds_unalign = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"{cfg.dataset}_llava-v1.5-13b_ds_unalign.pkl"
            )
        )
        unalign = parse_llava_yes_no(ds_unalign)
    elif shuffle_level == "class":
        class_unalign = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"{cfg.dataset}_llava-v1.5-13b_class_unalign.pkl"
            )
        )
        unalign = parse_llava_yes_no(class_unalign)
    elif shuffle_level == "object":
        obj_unalign = joblib.load(
            Path(
                cfg_dataset.paths.save_path
                + f"{cfg.dataset}_llava-v1.5-13b_obj_unalign.pkl"
            )
        )
        unalign = parse_llava_yes_no(obj_unalign)

    # print ROC
    print("Aligned vs Unaligned. Level: ", shuffle_level)
    tpr, fpr = boolean_binary_detection(align, unalign)
    print(f"tpr: {tpr}, fpr: {fpr}")
    return (fpr, tpr)


def llava_mislabeled_align(cfg: DictConfig) -> tuple[float, float]:
    """Return llava's mislabeled answer."""
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    llava_results = joblib.load(
        Path(cfg_dataset.paths.save_path + f"{cfg.dataset}_llava-v1.5-13b_aligned.pkl")
    )
    llava_results = parse_llava_yes_no(llava_results)
    wrong_labels_bool = parse_wrong_label(cfg)

    train_idx, val_idx = get_train_test_split_index(
        cfg.train_test_ratio, data1.shape[0]
    )
    train_wrong_labels_bool, val_wrong_labels_bool = train_test_split(
        wrong_labels_bool, train_idx, val_idx
    )
    train_llava_results, val_llava_results = train_test_split(
        llava_results, train_idx, val_idx
    )

    # separate aligned data and unaligned data
    val_llava_results_align = val_llava_results[~val_wrong_labels_bool]
    val_llava_results_unalign = val_llava_results[val_wrong_labels_bool]

    # print ROC
    print("Mislabeled vs correctly labeled.")
    tpr, fpr = boolean_binary_detection(
        val_llava_results_align, val_llava_results_unalign
    )
    print(f"tpr: {tpr}, fpr: {fpr}")
    return (fpr, tpr)


if __name__ == "__main__":

    @hydra.main(version_base=None, config_path="../config", config_name="main")
    def main(cfg: DictConfig) -> None:
        """Main function to query llava and save the aligned answer as pickle file.

        Args:
            cfg: config file
        """
        llava_shuffle_align(cfg=cfg)
        llava_mislabeled_align(cfg=cfg)

    main()
