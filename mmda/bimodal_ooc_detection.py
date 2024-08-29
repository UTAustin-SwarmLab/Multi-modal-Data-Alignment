"""This script is used to detect out-of-context in COSMOS."""

from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig

import hydra
from mmda.exps.hier_ooc import asif_hier_ooc, cca_hier_ooc, clip_like_hier_ooc
from mmda.exps.llava_alignment import llava_ooc_detection
from mmda.utils.roc_utils import cal_auc


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main function to generate the ROC curves of detecting out-of-context captions and images.

    Args:
        cfg: config file
    """
    assert cfg.dataset == "cosmos", f"{cfg.dataset} is not an ooc dataset."
    cfg_dataset = cfg[cfg.dataset]
    cca_roc_points = cca_hier_ooc(cfg)
    clip_roc_points = clip_like_hier_ooc(cfg)
    asif_roc_points = asif_hier_ooc(cfg)

    # plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot(
        [x[0] for x in cca_roc_points],
        [x[1] for x in cca_roc_points],
        "o-",
        ms=6,
        label=f"CSA (ours). AUC={cal_auc(cca_roc_points):.2f}",
        color="blue",
    )
    ax.plot(
        [x[0] for x in clip_roc_points],
        [x[1] for x in clip_roc_points],
        "^-",
        ms=6,
        label=f"CLIP. AUC={cal_auc(clip_roc_points):.2f}",
        color="red",
    )
    ax.plot(
        [x[0] for x in asif_roc_points],
        [x[1] for x in asif_roc_points],
        "D-",
        ms=6,
        label=f"ASIF. AUC={cal_auc(asif_roc_points):.2f}",
        color="green",
    )
    llava_fpr, llava_tpr = llava_ooc_detection(cfg)  # llava
    ax.plot(llava_fpr, llava_tpr, "x", ms=12, mew=3, label="LLaVA", c="black")
    ax.plot(0.26, 0.74, "P", ms=12, mew=3, label="COSMOS", c="darkorange")  # cosmos
    ax.set_xlabel("False positive rate", fontsize=16)
    ax.set_ylabel("True positive rate", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid()

    plots_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"hier_ooc_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}_{cfg_dataset.detection_rule}/"
    )
    plots_path.mkdir(parents=True, exist_ok=True)
    eq_label = "_noweight" if cfg[cfg.dataset].equal_weights else ""
    plt.tight_layout()
    fig.savefig(plots_path / f"ROC_ooc_dim{cfg[cfg.dataset].sim_dim}{eq_label}.png")


if __name__ == "__main__":
    main()
