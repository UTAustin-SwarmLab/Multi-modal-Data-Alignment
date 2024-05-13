"""This script is used to generate the ROC curves of detecting text description misalignment."""

from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig

import hydra
from mmda.exps.llava_alignment import llava_shuffle_align
from mmda.exps.shuffle_align import (
    asif_data_align,
    cca_data_align,
    clip_like_data_align,
)
from mmda.utils.roc_utils import cal_auc


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: PLR0915
    """Main function to generate the ROC curves of detecting text description misalignment.

    Args:
        cfg: config file
    """
    num_train_data = int(cfg.dataset_size[cfg.dataset] * cfg.train_test_ratio)
    clip_model_name = "CLAP" if cfg.dataset == "musiccaps" else "CLIP"
    print("number of training data", num_train_data)

    # plot the ROC curve
    fig, ax = plt.subplots()

    # dataset level shuffle ROC curve
    if cfg.dataset in cfg.dataset_level_datasets:
        roc_points = cca_data_align(cfg, "dataset")
        ds_auc = cal_auc(roc_points)
        clip_roc_ds_points = clip_like_data_align(cfg, "dataset")
        clip_ds_auc = cal_auc(clip_roc_ds_points)
        asif_roc_points = asif_data_align(cfg, "dataset")
        asif_ds_auc = cal_auc(asif_roc_points)
        ax.plot(
            [x[0] for x in roc_points],
            [x[1] for x in roc_points],
            "o-",
            ms=8,
            label=f"Random shuffle (ours). AUC={ds_auc:.3f}",
            color="blue",
        )
        ax.plot(
            [x[0] for x in clip_roc_ds_points],
            [x[1] for x in clip_roc_ds_points],
            "^-",
            ms=8,
            label=f"Random shuffle ({clip_model_name}). AUC={clip_ds_auc:.3f}",
            color="blue",
        )
        ax.plot(
            [x[0] for x in asif_roc_points],
            [x[1] for x in asif_roc_points],
            "D-",
            ms=8,
            label=f"ASIF. AUC={asif_ds_auc:.3f}",
            color="blue",
        )
        # LLaVA
        if cfg.dataset in cfg.shuffle_llava_datasets:
            llava_fpr, llava_tpr = llava_shuffle_align(cfg, "dataset")
            ax.plot(
                llava_fpr,
                llava_tpr,
                "x",
                ms=12,
                mew=3,
                label="LLaVA random shuffle.",
                c="blue",
            )

    # class level shuffle ROC curve
    if cfg.dataset in cfg.class_level_datasets:
        roc_class_points = cca_data_align(cfg, "class")
        class_auc = cal_auc(roc_class_points)
        clip_roc_class_points = clip_like_data_align(cfg, "class")
        clip_class_auc = cal_auc(clip_roc_class_points)
        asif_roc_class_points = asif_data_align(cfg, "class")
        asif_class_auc = cal_auc(asif_roc_class_points)
        ax.plot(
            [x[0] for x in roc_class_points],
            [x[1] for x in roc_class_points],
            "o-",
            ms=8,
            label=f"Class level shuffle (ours). AUC={class_auc:.3f}",
            color="red",
        )
        ax.plot(
            [x[0] for x in clip_roc_class_points],
            [x[1] for x in clip_roc_class_points],
            "^-",
            ms=8,
            label=f"Class level shuffle ({clip_model_name}). AUC={clip_class_auc:.3f}",
            color="red",
        )
        ax.plot(
            [x[0] for x in asif_roc_class_points],
            [x[1] for x in asif_roc_class_points],
            "D-",
            ms=8,
            label=f"ASIF. AUC={asif_class_auc:.3f}",
            color="red",
        )
        # LLAVA
        if cfg.dataset in cfg.shuffle_llava_datasets:
            llava_fpr, llava_tpr = llava_shuffle_align(cfg, "class")
            ax.plot(
                llava_fpr,
                llava_tpr,
                "x",
                ms=12,
                mew=3,
                label="LLaVA class level shuffle.",
                c="red",
            )

    # obj shuffle levels
    if cfg.dataset in cfg.object_level_datasets:
        # object level shuffle ROC curve
        roc_obj_points = cca_data_align(cfg, "object")
        obj_auc = cal_auc(roc_obj_points)
        clip_obj_roc_points = clip_like_data_align(cfg, "object")
        clip_obj_auc = cal_auc(clip_obj_roc_points)
        asif_obj_roc_points = asif_data_align(cfg, "object")
        asif_obj_auc = cal_auc(asif_obj_roc_points)
        ax.plot(
            [x[0] for x in roc_obj_points],
            [x[1] for x in roc_obj_points],
            "o-",
            ms=8,
            label=f"Object level shuffle (ours). AUC={obj_auc:.3f}",
            color="green",
        )
        ax.plot(
            [x[0] for x in clip_obj_roc_points],
            [x[1] for x in clip_obj_roc_points],
            "^-",
            ms=8,
            label=f"Object level shuffle ({clip_model_name}). AUC={clip_obj_auc:.3f}",
            color="green",
        )
        ax.plot(
            [x[0] for x in asif_obj_roc_points],
            [x[1] for x in asif_obj_roc_points],
            "D-",
            ms=8,
            label=f"ASIF. AUC={asif_obj_auc:.3f}",
            color="green",
        )
        # LLAVA
        if cfg.dataset in cfg.shuffle_llava_datasets:
            llava_fpr, llava_tpr = llava_shuffle_align(cfg, "object")
            ax.plot(
                llava_fpr,
                llava_tpr,
                "x",
                ms=12,
                mew=3,
                label="LLaVA object level shuffle.",
                c="green",
            )

    ax.set_title("ROC Curves of Detecting Modality Alignment")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right")
    ax.grid()
    eq_label = "_noweight" if cfg[cfg.dataset].equal_weights else ""
    fig.savefig(
        Path(
            cfg[cfg.dataset].paths.plots_path,
            f"shuffle_align_{cfg[cfg.dataset].text_encoder}_{cfg[cfg.dataset].img_encoder}",
            f"ROC_curves_size{num_train_data}_dim{cfg[cfg.dataset].sim_dim}{eq_label}.png",
        )
    )


if __name__ == "__main__":
    main()
