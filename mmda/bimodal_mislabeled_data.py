"""This script is used to detect mislabeled data in the bimodal datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig

import hydra
from mmda.exps.llava_alignment import llava_mislabeled_align
from mmda.exps.mislabel_align import (
    asif_detect_mislabeled_data,
    cca_detect_mislabeled_data,
    clip_like_detect_mislabeled_data,
)
from mmda.utils.roc_utils import cal_auc


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main function to generate the ROC curves of detecting mislabeled data.

    Args:
        cfg: config file
    """
    num_train_data = int(cfg.dataset_size[cfg.dataset] * cfg.train_test_ratio)
    clip_model_name = "CLAP" if cfg.dataset == "musiccaps" else "CLIP"
    print("number of training data", num_train_data)
    # plot the ROC curve
    fig, ax = plt.subplots()

    # dataset level shuffle ROC curve
    roc_points = cca_detect_mislabeled_data(cfg)
    auc = cal_auc(roc_points)
    clip_roc_ds_points = clip_like_detect_mislabeled_data(cfg)
    clip_auc = cal_auc(clip_roc_ds_points)
    asif_roc_points = asif_detect_mislabeled_data(cfg)
    asif_auc = cal_auc(asif_roc_points)
    ax.plot(
        [x[0] for x in roc_points],
        [x[1] for x in roc_points],
        "o-",
        ms=8,
        label=f"CSA (ours). AUC={auc:.2f}",
        color="blue",
    )
    ax.plot(
        [x[0] for x in clip_roc_ds_points],
        [x[1] for x in clip_roc_ds_points],
        "^-",
        ms=8,
        label=f"{clip_model_name}. AUC={clip_auc:.2f}",
        color="red",
    )
    ax.plot(
        [x[0] for x in asif_roc_points],
        [x[1] for x in asif_roc_points],
        "D-",
        ms=8,
        label=f"ASIF. AUC={asif_auc:.2f}",
        color="green",
    )

    if cfg.dataset in cfg.mislabel_llava_datasets:
        # plot LLaVA result.
        llava_fpr, llava_tpr = llava_mislabeled_align(cfg)
        ax.plot(llava_fpr, llava_tpr, "x", ms=12, mew=3, label="LLaVA", c="black")

    ax.set_xlabel("False positive rate", fontsize=16)
    ax.set_ylabel("True positive rate", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid()

    ds_label = "" if cfg.noisy_train_set else "_clean"
    eq_label = "_noweight" if cfg[cfg.dataset].equal_weights else ""
    plt.tight_layout()
    fig.savefig(
        Path(
            cfg[cfg.dataset].paths.plots_path,
            f"mislabeled_{cfg[cfg.dataset].text_encoder}_{cfg[cfg.dataset].img_encoder}",
            f"ROC_mislabeled_curves_size{num_train_data}_dim{cfg[cfg.dataset].sim_dim}{eq_label}{ds_label}.png",
        )
    )


if __name__ == "__main__":
    main()
