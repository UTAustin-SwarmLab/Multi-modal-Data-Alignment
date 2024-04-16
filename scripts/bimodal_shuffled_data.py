import os

import matplotlib.pyplot as plt
from omegaconf import DictConfig

import hydra
from mmda.data_shuffle_align import (
    ASIF_data_align,
    CCA_data_align,
    CLIP_like_data_align,
)
from mmda.utils.dataset_utils import load_dataset_config
from mmda.utils.sim_utils import (
    cal_AUC,
)
from scripts.parse_llava_alignment_result import llava_shuffle_align


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):  # noqa: D103
    if cfg.dataset == "musiccaps":
        num_train_data = int(5397 * cfg.train_test_ratio)
        clip_model_name = "CLAP"
    elif cfg.dataset == "sop":
        num_train_data = int(56222 * cfg.train_test_ratio)
        clip_model_name = "CLIP"
    elif cfg.dataset == "imagenet":
        num_train_data = int(50_000 * cfg.train_test_ratio)
        clip_model_name = "CLIP"
    elif cfg.dataset == "tiil":
        num_train_data = int(7138 * 2 * cfg.train_test_ratio)
        clip_model_name = "CLIP"
    elif cfg.dataset == "pitts":
        num_train_data = int(17608 * cfg.train_test_ratio)
        clip_model_name = "CLIP"
    # TODO: add more datasets
    else:
        raise ValueError(f"Dataset {cfg.dataset} not supported.")
    print("number of training data", num_train_data)

    cfg_dataset = load_dataset_config(cfg)

    # plot the ROC curve
    fig, ax = plt.subplots()

    # dataset level shuffle ROC curve
    if cfg.dataset in cfg.dataset_level_datasets:
        roc_points = CCA_data_align(cfg, "dataset")
        ds_auc = cal_AUC(roc_points)
        clip_roc_ds_points = CLIP_like_data_align(cfg, "dataset")
        clip_ds_auc = cal_AUC(clip_roc_ds_points)
        asif_roc_points = ASIF_data_align(cfg, "dataset")
        asif_ds_auc = cal_AUC(asif_roc_points)
        ax.plot(
            [x[0] for x in roc_points],
            [x[1] for x in roc_points],
            "o-",
            label=f"Random shuffle (ours). AUC={ds_auc:.3f}",
            color="blue",
        )
        ax.plot(
            [x[0] for x in clip_roc_ds_points],
            [x[1] for x in clip_roc_ds_points],
            "+-",
            label=f"Random shuffle ({clip_model_name}). AUC={clip_ds_auc:.3f}",
            color="blue",
        )
        ax.plot(
            [x[0] for x in asif_roc_points],
            [x[1] for x in asif_roc_points],
            "D-",
            label=f"ASIF. AUC={asif_ds_auc:.3f}",
            color="blue",
        )
        # LLaVA
        if cfg.dataset in cfg.llava_datasets:
            llava_FPR, llava_TPR = llava_shuffle_align(cfg, "dataset")
            ax.plot(llava_FPR, llava_TPR, "x", ms=12, mew=3, label="LLaVA random shuffle.", c="blue")

    # class level shuffle ROC curve
    if cfg.dataset in cfg.class_level_datasets:
        roc_class_points = CCA_data_align(cfg, "class")
        class_auc = cal_AUC(roc_class_points)
        clip_roc_class_points = CLIP_like_data_align(cfg, "class")
        clip_class_auc = cal_AUC(clip_roc_class_points)
        asif_roc_class_points = ASIF_data_align(cfg, "class")
        asif_class_auc = cal_AUC(asif_roc_class_points)
        ax.plot(
            [x[0] for x in roc_class_points],
            [x[1] for x in roc_class_points],
            "o-",
            label=f"Class level shuffle (ours). AUC={class_auc:.3f}",
            color="red",
        )
        ax.plot(
            [x[0] for x in clip_roc_class_points],
            [x[1] for x in clip_roc_class_points],
            "+-",
            label=f"Class level shuffle ({clip_model_name}). AUC={clip_class_auc:.3f}",
            color="red",
        )
        ax.plot(
            [x[0] for x in asif_roc_class_points],
            [x[1] for x in asif_roc_class_points],
            "D-",
            label=f"ASIF. AUC={asif_class_auc:.3f}",
            color="red",
        )
        # LLAVA
        if cfg.dataset in cfg.llava_datasets:
            llava_FPR, llava_TPR = llava_shuffle_align(cfg, "class")
            ax.plot(llava_FPR, llava_TPR, "x", ms=12, mew=3, label="LLaVA class level shuffle.", c="red")

    # obj shuffle levels
    if cfg.dataset in cfg.object_level_datasets:
        # object level shuffle ROC curve
        roc_obj_points = CCA_data_align(cfg, "object")
        obj_auc = cal_AUC(roc_obj_points)
        clip_obj_roc_points = CLIP_like_data_align(cfg, "object")
        clip_obj_auc = cal_AUC(clip_obj_roc_points)
        asif_obj_roc_points = ASIF_data_align(cfg, "object")
        asif_obj_auc = cal_AUC(asif_obj_roc_points)
        ax.plot(
            [x[0] for x in roc_obj_points],
            [x[1] for x in roc_obj_points],
            "o-",
            label=f"Object level shuffle (ours). AUC={obj_auc:.3f}",
            color="green",
        )
        ax.plot(
            [x[0] for x in clip_obj_roc_points],
            [x[1] for x in clip_obj_roc_points],
            "+-",
            label=f"Object level shuffle ({clip_model_name}). AUC={clip_obj_auc:.3f}",
            color="green",
        )
        ax.plot(
            [x[0] for x in asif_obj_roc_points],
            [x[1] for x in asif_obj_roc_points],
            "D-",
            label=f"ASIF. AUC={asif_obj_auc:.3f}",
            color="green",
        )
        # LLAVA
        if cfg.dataset in cfg.llava_datasets:
            llava_FPR, llava_TPR = llava_shuffle_align(cfg, "object")
            ax.plot(llava_FPR, llava_TPR, "x", ms=12, mew=3, label="LLaVA object level shuffle.", c="green")

    ax.set_title("ROC Curves of Detecting Modality Alignment")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right")
    ax.grid()
    if cfg_dataset.equal_weights:
        fig.savefig(
            os.path.join(
                cfg_dataset.paths.plots_path,
                f"shuffle_align/ROC_curves_size{num_train_data}_dim{cfg_dataset.sim_dim}_noweight.png",
            )
        )
    else:
        fig.savefig(
            os.path.join(
                cfg_dataset.paths.plots_path,
                f"shuffle_align/ROC_curves_size{num_train_data}_dim{cfg_dataset.sim_dim}.png",
            )
        )
    return


if __name__ == "__main__":
    main()
