"""Plot functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import DictConfig

import hydra


@hydra.main(version_base=None, config_path="../config", config_name="main")
def plot_single_modal_recall(cfg: DictConfig) -> None:
    """Plot single-modal recall."""
    cell_size = 30
    label_size = 30
    ticks_size = 28

    cfg_dataset = cfg["KITTI"]
    dir_path = Path(cfg_dataset.paths.plots_path)
    single1_recalls = [[31.9, 31.9, 31.7], [32.4, 32.4, 31.8], [33.7, 32.8, 32.2]]
    single_recalls = np.array(single1_recalls).reshape(3, 3)
    plt.figure(figsize=(9, 9))
    ax = sns.heatmap(
        single_recalls,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=False,
        square=True,
        xticklabels=["Image", "Lidar", "Text"],
        yticklabels=["Image", "Lidar", "Text"],
        annot=True,
        annot_kws={"size": cell_size, "weight": "bold"},
    )
    ax.xaxis.tick_top()
    plt.xlabel("Reference modality", fontsize=label_size)
    plt.ylabel("Query modality", fontsize=label_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout()
    ax.xaxis.set_label_position("top")  # Move the label to the top
    plt.subplots_adjust(bottom=-0.05)
    plt.savefig(
        dir_path
        / f"single_modal_recall5_{cfg_dataset.retrieval_dim}_{cfg_dataset.mask_ratio}.pdf"
    )

    dir_path = Path("plots/KITTI/")
    single1_recalls = [[32.4], [32.8]]
    single_recalls = np.array(single1_recalls).reshape(2, 1)
    plt.figure(figsize=(6, 8))
    ax = sns.heatmap(
        single_recalls,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=False,
        square=True,
        xticklabels=["LiDAR (Lip-loc)"],
        yticklabels=["LiDAR (Lip-loc)", "Text (GTR)"],
        annot=True,
        annot_kws={"size": cell_size, "weight": "bold"},
    )
    ax.xaxis.tick_top()
    plt.xlabel("Reference modality", fontsize=label_size)
    plt.ylabel("Query modality", fontsize=label_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    ax.xaxis.set_label_position("top")  # Move the label to the top
    plt.tight_layout()
    plt.savefig(dir_path / f"bimodal_recall5_{cfg_dataset.retrieval_dim}.pdf")

    cfg_dataset = cfg["MSRVTT"]
    dir_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"{cfg_dataset.img_encoder}_{cfg_dataset.audio_encoder}"
    )
    single1_recalls = [49.3, 2.6]
    single_recalls = np.array(single1_recalls).reshape(1, 2)
    plt.figure(figsize=(6, 4.5))
    ax = sns.heatmap(
        single_recalls,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=False,
        square=True,
        xticklabels=["Image", "Audio"],
        yticklabels=["Text"],
        annot=True,
        annot_kws={"size": cell_size, "weight": "bold"},
    )
    ax.xaxis.tick_top()
    plt.xlabel("Reference modality", fontsize=label_size)
    plt.ylabel("Query modality", fontsize=label_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout()
    ax.xaxis.set_label_position("top")  # Move the label to the top
    plt.savefig(
        dir_path
        / f"single_modal_recall5_{cfg_dataset.retrieval_dim}_{cfg_dataset.mask_ratio}.pdf"
    )

    cfg_dataset = cfg["BTC"]
    dir_path = Path(cfg_dataset.paths.plots_path)
    single1_recalls = [[4.1, 4.7], [3.4, 4.7]]
    single_recalls = np.array(single1_recalls).reshape(2, 2)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(
        single_recalls,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=False,
        square=True,
        xticklabels=["Time", "Stats"],
        yticklabels=["Prev News", "Text (2)"],
        annot=True,
        annot_kws={"size": cell_size, "weight": "bold"},
    )
    ax.xaxis.tick_top()
    plt.xlabel("Reference modality", fontsize=label_size)
    plt.ylabel("Query modality", fontsize=label_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout()
    ax.xaxis.set_label_position("top")  # Move the label to the top
    plt.subplots_adjust(bottom=-0.05)
    plt.savefig(
        dir_path
        / f"single_modal_recall5_{cfg_dataset.retrieval_dim}_{cfg_dataset.mask_ratio}.pdf"
    )


if __name__ == "__main__":
    plot_single_modal_recall()
