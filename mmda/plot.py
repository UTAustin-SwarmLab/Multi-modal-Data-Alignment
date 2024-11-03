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
    cfg_dataset = cfg[cfg.dataset]
    dir_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"{cfg_dataset.img_encoder}_{cfg_dataset.audio_encoder}"
    )
    single1_recalls = [49.3, 2.6]
    single_recalls = np.array(single1_recalls).reshape(1, 2)
    plt.figure(figsize=(6, 4.3))
    ax = sns.heatmap(
        single_recalls,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=False,
        square=True,
        xticklabels=["Image", "Audio"],
        yticklabels=["Text"],
        annot=True,
        annot_kws={"size": 26, "weight": "bold"},
    )
    ax.xaxis.tick_top()
    plt.xlabel("Reference modality", fontsize=30)
    plt.ylabel("Query modality", fontsize=30)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    ax.xaxis.set_label_position("top")  # Move the label to the top
    plt.savefig(
        dir_path
        / f"single_modal_recall5_{cfg_dataset.retrieval_dim}_{cfg_dataset.mask_ratio}.pdf"
    )
    print(f"Single-modal recall plot saved to {dir_path}")


if __name__ == "__main__":
    plot_single_modal_recall()
