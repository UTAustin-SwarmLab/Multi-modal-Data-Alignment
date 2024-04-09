import os

import matplotlib.pyplot as plt
from omegaconf import DictConfig

import hydra
from mmda.data_real_align import (
    ASIF_detect_mislabeled_data,
    CCA_detect_mislabeled_data,
    CLIP_like_detect_mislabeled_data,
)
from mmda.utils.data_utils import (
    load_two_encoder_data,
)
from mmda.utils.sim_utils import (
    cal_AUC,
)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):  # noqa: D103
    if cfg.dataset == "imagenet":
        num_train_data = int(50_000 * cfg.train_test_ratio)
        clip_model_name = "CLIP"
    else:
        raise ValueError(f"Dataset {cfg.dataset} not supported.")
    print("number of training data", num_train_data)

    cfg_dataset, _, _ = load_two_encoder_data(cfg)

    # plot the ROC curve
    fig, ax = plt.subplots()

    # dataset level shuffle ROC curve
    roc_points = CCA_detect_mislabeled_data(cfg)
    auc = cal_AUC(roc_points)
    clip_roc_ds_points = CLIP_like_detect_mislabeled_data(cfg)
    clip_auc = cal_AUC(clip_roc_ds_points)
    asif_roc_points = ASIF_detect_mislabeled_data(cfg)
    asif_auc = cal_AUC(asif_roc_points)
    ax.plot(
        [x[0] for x in roc_points],
        [x[1] for x in roc_points],
        "o-",
        label=f"Random shuffle (ours). AUC={auc:.3f}",
        color="blue",
    )
    ax.plot(
        [x[0] for x in clip_roc_ds_points],
        [x[1] for x in clip_roc_ds_points],
        "+-",
        label=f"Random shuffle ({clip_model_name}). AUC={clip_auc:.3f}",
        color="blue",
    )
    ax.plot(
        [x[0] for x in asif_roc_points],
        [x[1] for x in asif_roc_points],
        "D-",
        label=f"ASIF. AUC={asif_auc:.3f}",
        color="blue",
    )

    ax.set_title("ROC Curves of Detecting Mislabeled Data")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right")
    ax.grid()

    train_label = "" if cfg.noisy_train_set else "_clean"
    eq_label = "_noweight" if cfg_dataset.equal_weights else ""
    fig.savefig(
        os.path.join(
            cfg_dataset.paths.plots_path,
            f"mislabeled/ROC_mislabeled_curves_size{num_train_data}_dim{cfg_dataset.sim_dim}{eq_label}{train_label}.png",
        )
    )
    return


if __name__ == "__main__":
    main()
