import pickle
from typing import List

import numpy as np
from omegaconf import DictConfig

from mmda.utils.hydra_utils import hydra_main


def parse_llava_yes_no(llava_output: List[str]) -> np.ndarray:
    """
    Parse the llava output to see if the answer is yes or no
    :param llava_output
    :return: boolean array
    """
    yes_no_array = []
    for path_answer in llava_output[0]:
        if "yes" in path_answer[1].lower():
            yes_no_array.append(True)
        else:
            yes_no_array.append(False)
    yes_no_array = np.array(yes_no_array).astype(bool)
    return yes_no_array

def boolean_binary_detection(align: np.ndarray, unalign: np.ndarray) -> List[List[float]]:
    """
    Calculate the ROC points for boolean data
    :param align: boolean outcome of aligned data
    :param unalign: boolean outcome of unaligned data
    :param unalign_gt: ground truth of unaligned data
    :return: ROC points
    """
    TP = np.sum(align)
    FP = np.sum(unalign)
    FN = np.sum(1 - align)
    TN = np.sum(1 - unalign)
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR

@hydra_main(version_base=None, config_path='../config', config_name='sop')
def SOP_llava_align(cfg: DictConfig):
    # set random seed
    np.random.seed(cfg.seed)

    # load image embeddings and text embeddings
    with open(cfg.paths.save_path + "sop_llava-v1.5-13b_aligned.pkl", 'rb') as f:
        align = pickle.load(f)
    with open(cfg.paths.save_path + "sop_llava-v1.5-13b_ds_unalign.pkl", 'rb') as f:
        ds_unalign = pickle.load(f)
    with open(cfg.paths.save_path + "sop_llava-v1.5-13b_class_unalign.pkl", 'rb') as f:
        class_unalign = pickle.load(f)
    with open(cfg.paths.save_path + "sop_llava-v1.5-13b_obj_unalign.pkl", 'rb') as f:
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

if __name__ == '__main__':
    SOP_llava_align()
