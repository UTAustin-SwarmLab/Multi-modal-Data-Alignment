import os

import matplotlib.pyplot as plt
from sop_class_image_text_align import SOP_class_align
from sop_image_text_align import SOP_align
from sop_obj_image_text_align import SOP_obj_align

from hydra.core.global_hydra import GlobalHydra
from tife.utils.sim_utils import cal_AUC


def main():
    train_test_ratio = 0.7
    num_train_data = 56222 * train_test_ratio
    print(num_train_data)

    print("Start to calculate the ROC curve of detecting modality alignment.")
    ds_roc_points = SOP_align()
    ds_auc = cal_AUC(ds_roc_points)
    GlobalHydra.instance().clear()

    print("Start to calculate the ROC curve of detecting modality alignment with class level shuffle.")
    class_roc_points = SOP_class_align()
    class_auc = cal_AUC(class_roc_points)
    GlobalHydra.instance().clear()

    print("Start to calculate the ROC curve of detecting modality alignment with object level shuffle.")
    obj_roc_points = SOP_obj_align()
    obj_auc = cal_AUC(obj_roc_points)
    GlobalHydra.instance().clear()

    # plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in ds_roc_points], [x[1] for x in ds_roc_points], 'o-', label=f'Random shuffle. AUC={ds_auc:.3f}', color='blue')
    ax.plot([x[0] for x in class_roc_points], [x[1] for x in class_roc_points], 'o-', label=f'Class level shuffle. AUC={class_auc:.3f}', color='red')
    ax.plot([x[0] for x in obj_roc_points], [x[1] for x in obj_roc_points], 'o-', label=f'Object level shuffle. AUC={obj_auc:.3f}', color='green')
    ax.plot([0.02158], [0.97213], 'x', label='LLaVA random shuffle.', color='blue')
    ax.plot([0.14543], [0.97213], 'x', label='LLaVA class level shuffle.', color='red')
    ax.plot([0.78223], [0.97213], 'x', label='LLaVA object level shuffle.', color='green')
    ax.set_title('ROC Curves of Detecting Modality Alignment')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(os.path.dirname(__file__), "./plots/") + f'roc_curves_size{num_train_data}.png')
    return

if __name__ == "__main__":
    main()