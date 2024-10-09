"""This script is used to retrieve multimodal datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

import hydra
from mmda.exps.any2any_retrieval import any2any_retrieval


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main function to generate the retrieval results of the multimodal datasets.

    Args:
        cfg: config file
    """
    cfg_dataset = cfg[cfg.dataset]
    assert (
        cfg.dataset in cfg.any_retrieval_datasets
    ), f"{cfg.dataset} is not for any2any retrieval."
    (
        (maps, precisions, recalls),
        (full_maps, full_precisions, full_recalls),
        (single1_aps, single1_precisions, single1_recalls),
    ) = any2any_retrieval(cfg)

    # write the results to a csv file
    data = {
        "method": [
            "Conformal Retrieval (Missing)",
            "Conformal Retrieval (Full)",
        ],
        "mAP@5": [maps[5], full_maps[5]],
        "mAP@20": [maps[20], full_maps[20]],
        "Precision@1": [precisions[1], full_precisions[1]],
        "Precision@5": [precisions[5], full_precisions[5]],
        "Precision@20": [precisions[20], full_precisions[20]],
        "Recall@1": [recalls[1], full_recalls[1]],
        "Recall@5": [recalls[5], full_recalls[5]],
        "Recall@20": [recalls[20], full_recalls[20]],
    }
    df = pd.DataFrame(data)
    df_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"any2any_retrieval_{cfg_dataset.retrieval_dim}_{cfg_dataset.mask_ratio}.csv"
    )
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(df_path, index=False)

    # plot heatmap of single modality retrieval
    single1_recalls = np.array(list(single1_recalls.values())).reshape(3, 3) * 100
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        single1_recalls,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=False,
        square=True,
        xticklabels=["Image", "Lidar", "Text"],
        yticklabels=["Image", "Lidar", "Text"],
        annot=True,
        annot_kws={"size": 18, "weight": "bold"},
    )
    ax.xaxis.tick_top()
    plt.xlabel("Reference modality", fontsize=20)
    plt.ylabel("Query modality", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.xaxis.set_label_position("top")  # Move the label to the top
    plt.savefig(
        Path(cfg_dataset.paths.plots_path)
        / f"single_modal_recall5_{cfg_dataset.retrieval_dim}_{cfg_dataset.mask_ratio}.png"
    )


if __name__ == "__main__":
    main()
# poetry run python mmda/any2any_conformal_retrieval.py
