"""Plot functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import DictConfig

import hydra


@hydra.main(version_base=None, config_path="../config", config_name="main")
def plot_kitti_cross_retrieval(cfg: DictConfig) -> None:
    """Plot cross-modal retrieval results for KITTI dataset.

    Args:
        cfg: Configuration object containing dataset parameters
    """
    cell_size = 30
    label_size = 30
    ticks_size = 28
    cfg_dataset = cfg["KITTI"]
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
        annot_kws={"size": cell_size + 10, "weight": "bold"},
    )
    ax.xaxis.tick_top()
    plt.xlabel("Reference modality", fontsize=label_size)
    plt.ylabel("Query modality", fontsize=label_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    ax.xaxis.set_label_position("top")  # Move the label to the top
    plt.tight_layout()
    plt.savefig(dir_path / f"bimodal_recall5_{cfg_dataset.retrieval_dim}.pdf")


if __name__ == "__main__":
    plot_kitti_cross_retrieval()
