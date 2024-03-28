import matplotlib.pyplot as plt
from omegaconf import DictConfig
from sop_class_image_text_align import SOP_class_align, SOP_CLIP_class_align
from sop_image_text_align import SOP_align, SOP_CLIP_align
from sop_obj_image_text_align import SOP_CLIP_obj_align, SOP_obj_align

from mmda.utils.hydra_utils import hydra_main
from mmda.utils.sim_utils import cal_AUC


@hydra_main(version_base=None, config_path='../config', config_name='sop')
def main(cfg: DictConfig):
    train_test_ratio = 0.7
    num_train_data = int(56222 * train_test_ratio)
    print(num_train_data)

    print("Start to calculate the ROC curve of detecting modality alignment.")
    ds_roc_points = SOP_align(cfg)
    ds_auc = cal_AUC(ds_roc_points)
    ds_cos_roc_points = SOP_CLIP_align(cfg)
    ds_cos_auc = cal_AUC(ds_cos_roc_points)

    print("Start to calculate the ROC curve of detecting modality alignment with class level shuffle.")
    class_roc_points = SOP_class_align(cfg)
    class_auc = cal_AUC(class_roc_points)
    class_cos_roc_points = SOP_CLIP_class_align(cfg)
    class_cos_auc = cal_AUC(class_cos_roc_points)

    print("Start to calculate the ROC curve of detecting modality alignment with object level shuffle.")
    obj_roc_points = SOP_obj_align(cfg)
    obj_auc = cal_AUC(obj_roc_points)
    obj_cos_roc_points = SOP_CLIP_obj_align(cfg)
    obj_cos_auc = cal_AUC(obj_cos_roc_points)

    # plot the ROC curve
    fig, ax = plt.subplots()

    # Ours
    ax.plot([x[0] for x in ds_roc_points], [x[1] for x in ds_roc_points], 'o-', label=f'Random shuffle. AUC={ds_auc:.3f}', color='blue')
    ax.plot([x[0] for x in class_roc_points], [x[1] for x in class_roc_points], 'o-', label=f'Class level shuffle. AUC={class_auc:.3f}', color='red')
    ax.plot([x[0] for x in obj_roc_points], [x[1] for x in obj_roc_points], 'o-', label=f'Object level shuffle. AUC={obj_auc:.3f}', color='green')
    # CLIP encoders
    ax.plot([x[0] for x in ds_cos_roc_points], [x[1] for x in ds_cos_roc_points], '+-', label=f'Random shuffle (CLIP). AUC={ds_cos_auc:.3f}', color='blue')
    ax.plot([x[0] for x in class_cos_roc_points], [x[1] for x in class_cos_roc_points], '+-', label=f'Class level shuffle (CLIP). AUC={class_cos_auc:.3f}', color='red')
    ax.plot([x[0] for x in obj_cos_roc_points], [x[1] for x in obj_cos_roc_points], '+-', label=f'Object level shuffle (CLIP). AUC={obj_cos_auc:.3f}', color='green')
    # LLaVA
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
    if cfg.equal_weights:
        fig.savefig(cfg.paths.plots_path + f'ROC_curves_size{num_train_data}_noweight.png')
    else:
        fig.savefig(cfg.paths.plots_path + f'ROC_curves_size{num_train_data}.png')
    return

if __name__ == "__main__":
    main()