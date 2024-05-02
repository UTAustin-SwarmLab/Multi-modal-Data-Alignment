"""This script is used to detect out-of-context in COSMOS."""

from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig

import hydra
from mmda.exps.hier_ooc import asif_hier_ooc, cca_hier_ooc, clip_like_hier_ooc
from mmda.utils.roc_utils import cal_auc, select_maximum_auc


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main function to generate the ROC curves of detecting out-of-context captions and images.

    Args:
        cfg: config file
    """
    assert cfg.dataset == "cosmos", f"{cfg.dataset} is not an ooc dataset."
    cfg_dataset = cfg[cfg.dataset]
    cca_tps = cca_hier_ooc(cfg)
    cca_roc_points = select_maximum_auc(cca_tps)

    # (txt_threshold, text_img_threshold): (tp, fp, fn, tn)
    clip_tps = clip_like_hier_ooc(cfg)
    clip_roc_points = select_maximum_auc(clip_tps)

    asif_tps = asif_hier_ooc(cfg)
    asif_roc_points = select_maximum_auc(asif_tps)

    # plot the ROC curve
    fig, ax = plt.subplots()
    ax.set_title("ROC Curves of Detecting Out-of-context Captions and Images")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.plot(
        [x[0] for x in cca_roc_points],
        [x[1] for x in cca_roc_points],
        "o-",
        ms=6,
        label=f"Ours. AUC={cal_auc(cca_roc_points):.3f}",
        color="blue",
    )
    ax.plot(
        [x[0] for x in clip_roc_points],
        [x[1] for x in clip_roc_points],
        "^-",
        ms=6,
        label=f"CLIP. AUC={cal_auc(clip_roc_points):.3f}",
        color="blue",
    )
    ax.plot(
        [x[0] for x in asif_roc_points],
        [x[1] for x in asif_roc_points],
        "D-",
        ms=6,
        label=f"ASIF. AUC={cal_auc(asif_roc_points):.3f}",
        color="blue",
    )
    ax.plot(0.26, 0.74, "x", ms=12, mew=3, label="COSMOS", c="blue")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right")
    ax.grid()

    plots_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"hier_ooc_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/"
    )
    plots_path.mkdir(parents=True, exist_ok=True)
    eq_label = "_noweight" if cfg[cfg.dataset].equal_weights else ""
    fig.savefig(plots_path / f"ROC_ooc_dim{cfg[cfg.dataset].sim_dim}{eq_label}.png")


if __name__ == "__main__":
    main()
