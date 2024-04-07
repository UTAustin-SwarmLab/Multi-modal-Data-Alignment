import pickle

import numpy as np
from omegaconf import DictConfig

import hydra


def parse_llava_yes_no(llava_output: list[str]) -> np.ndarray:
    """Parse the llava output to see if the answer is yes or no.

    Args:
        llava_output: lsit of llava output
    Returns:
        yes_no_array: numpy array of boolean values
    """
    yes_no_array = []
    for path_answer in llava_output[0]:
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
def SOP_llava_align(cfg: DictConfig):  # noqa: D103
    # set random seed
    np.random.seed(cfg.seed)

    # load image embeddings and text embeddings
    with open(cfg.paths.save_path + "sop_llava-v1.5-13b_aligned.pkl", "rb") as f:
        align = pickle.load(f)
    with open(cfg.paths.save_path + "sop_llava-v1.5-13b_ds_unalign.pkl", "rb") as f:
        ds_unalign = pickle.load(f)
    with open(cfg.paths.save_path + "sop_llava-v1.5-13b_class_unalign.pkl", "rb") as f:
        class_unalign = pickle.load(f)
    with open(cfg.paths.save_path + "sop_llava-v1.5-13b_obj_unalign.pkl", "rb") as f:
        obj_unalign = pickle.load(f)
    align = parse_llava_yes_no(align)
    ds_unalign = parse_llava_yes_no(ds_unalign)
    class_unalign = parse_llava_yes_no(class_unalign)
    obj_unalign = parse_llava_yes_no(obj_unalign)
    print(obj_unalign)

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


if __name__ == "__main__":
    SOP_llava_align()
