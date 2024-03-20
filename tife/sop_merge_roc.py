import os

import matplotlib.pyplot as plt
from sop_class_image_text_align import SOP_class_align
from sop_image_text_align import SOP_align
from sop_obj_image_text_align import SOP_obj_align

from hydra.core.global_hydra import GlobalHydra
from tife.utils.sim_utils import cal_AUC


def main():
    ds_roc_points = SOP_align()
    ds_auc = cal_AUC(ds_roc_points)
    GlobalHydra.instance().clear()

    class_roc_points = SOP_class_align()
    class_auc = cal_AUC(class_roc_points)
    GlobalHydra.instance().clear()

    obj_roc_points = SOP_obj_align()
    obj_auc = cal_AUC(obj_roc_points)
    GlobalHydra.instance().clear()

    # plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in ds_roc_points], [x[1] for x in ds_roc_points], 'o-', label=f'Random shuffle. AUC={ds_auc:.3f}', color='blue')
    ax.plot([x[0] for x in class_roc_points], [x[1] for x in class_roc_points], 'o-', label=f'Class level shuffle. AUC={class_auc:.3f}', color='red')
    ax.plot([x[0] for x in obj_roc_points], [x[1] for x in obj_roc_points], 'o-', label=f'Object level shuffle. AUC={obj_auc:.3f}', color='green')
    ax.plot([0], [0], 'o', label='LLaVA-v1.5-13B', color='black')
    ax.set_title('ROC Curves of Detecting Modality Alignment')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(os.path.dirname(__file__), "./plots/") + 'roc_curves.png')
    return

if __name__ == "__main__":
    main()