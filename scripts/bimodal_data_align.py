import matplotlib.pyplot as plt
from omegaconf import DictConfig

from mmda.data_align import (
    CCA_data_align,
    CLIP_like_data_align,
)
from mmda.utils.data_utils import (
    load_two_encoder_data,
)
from mmda.utils.hydra_utils import hydra_main
from mmda.utils.sim_utils import (
    cal_AUC,
)


@hydra_main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):  # noqa: D103
    if cfg.dataset == "musiccaps":
        num_train_data = int(5397 * cfg.train_test_ratio)
    elif cfg.dataset == "sop":
        num_train_data = int(56222 * cfg.train_test_ratio)
    else:
        raise ValueError(f"Dataset {cfg.dataset} not supported.")
    print("number of training data", num_train_data)

    cfg_dataset, _, _ = load_two_encoder_data(cfg)

    # plot different ROC curves for different shuffle levels (SOP only)
    if cfg.dataset == "sop":
        # dataset level shuffle ROC curve
        roc_points = CCA_data_align(cfg, "dataset")
        ds_auc = cal_AUC(roc_points)
        clip_roc_ds_points = CLIP_like_data_align(cfg, "dataset")
        clip_ds_auc = cal_AUC(clip_roc_ds_points)

        # class level shuffle ROC curve
        roc_class_points = CCA_data_align(cfg, "class")
        class_auc = cal_AUC(roc_class_points)
        clip_roc_class_points = CLIP_like_data_align(cfg, "class")
        clip_class_auc = cal_AUC(clip_roc_class_points)

        # object level shuffle ROC curve
        roc_obj_points = CCA_data_align(cfg, "object")
        obj_auc = cal_AUC(roc_obj_points)
        clip_obj_roc_points = CLIP_like_data_align(cfg, "object")
        clip_obj_auc = cal_AUC(clip_obj_roc_points)

    else:
        roc_points = CCA_data_align(cfg, "dataset")
        ds_auc = cal_AUC(roc_points)
        clip_roc_ds_points = CLIP_like_data_align(cfg, "dataset")
        clip_ds_auc = cal_AUC(clip_roc_ds_points)

    # plot the ROC curve
    fig, ax = plt.subplots()
    # Ours
    ax.plot(
        [x[0] for x in roc_points],
        [x[1] for x in roc_points],
        "o-",
        label=f"Random shuffle (ours). AUC={ds_auc:.3f}",
        color="blue",
    )
    # CLIP encoders
    ax.plot(
        [x[0] for x in clip_roc_ds_points],
        [x[1] for x in clip_roc_ds_points],
        "+-",
        label=f"Random shuffle (CLAP). AUC={clip_ds_auc:.3f}",
        color="blue",
    )
    # SOP only
    if cfg.dataset == "sop":
        # CCA and CLIP like models
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
            label=f"Class level shuffle (CLAP). AUC={clip_class_auc:.3f}",
            color="red",
        )
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
            label=f"Object level shuffle (CLAP). AUC={clip_obj_auc:.3f}",
            color="green",
        )
        # LLaVA
        ax.plot([0.02158], [0.97213], "x", markersize=12, mew=3, label="LLaVA random shuffle.", color="blue")
        ax.plot([0.14543], [0.97213], "x", markersize=12, mew=3, label="LLaVA class level shuffle.", color="red")
        ax.plot([0.78223], [0.97213], "x", markersize=12, mew=3, label="LLaVA object level shuffle.", color="green")

    ax.set_title("ROC Curves of Detecting Modality Alignment")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right")
    ax.grid()
    if cfg_dataset.equal_weights:
        fig.savefig(
            cfg_dataset.paths.plots_path + f"ROC_curves_size{num_train_data}_dim{cfg_dataset.sim_dim}_noweight.png"
        )
    else:
        fig.savefig(cfg_dataset.paths.plots_path + f"ROC_curves_size{num_train_data}_dim{cfg_dataset.sim_dim}.png")
    return


if __name__ == "__main__":
    main()
