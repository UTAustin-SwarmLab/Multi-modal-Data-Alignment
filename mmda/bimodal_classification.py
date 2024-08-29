"""This script is used to detect mislabeled data in the bimodal datasets."""

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import hydra
from mmda.exps.classification import (
    asif_classification,
    cca_classification,
    clip_like_classification,
)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main function to generate the classification results of the bimodal datasets.

    Args:
        cfg: config file
    """
    plot = True
    assert (
        cfg.dataset in cfg.classification_datasets
    ), f"{cfg.dataset} is not for classification."
    cfg_dataset = cfg[cfg.dataset]

    save_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"classify_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/"
        / "accuracy.csv"
    )
    for train_test_ratio in cfg_dataset.train_test_ratios:
        asif_accs = asif_classification(cfg, train_test_ratio)
        cca_accs = cca_classification(cfg, train_test_ratio)
        clip_accs = clip_like_classification(cfg, train_test_ratio)
        # write accuracy to file
        if not save_path.exists():
            with save_path.open("a") as f:
                f.write("train_test_ratio,cca_accs,clip_accs,asif_accs\n")
        with save_path.open("a") as f:
            f.write(f"{train_test_ratio},{cca_accs},{clip_accs},{asif_accs}\n")

    if plot and save_path.exists():
        df = pd.read_csv(save_path)
        print(df, df.columns)
        ratios = df["train_test_ratio"] * 50_000
        cca_accs = df["cca_accs"]
        clip_accs = df["clip_accs"]
        asif_accs = df["asif_accs"]
        fig, ax = plt.subplots()
        ax.plot(
            ratios,
            cca_accs,
            "o-",
            ms=6,
            label="CSA (ours)",
            color="blue",
        )
        ax.plot(
            ratios,
            clip_accs,
            "^-",
            ms=6,
            label="CLIP",
            color="red",
        )
        ax.plot(
            ratios,
            asif_accs,
            "D-",
            ms=6,
            label="ASIF",
            color="green",
        )
        ax.set_xlabel("Number of training data", fontsize=16)
        ax.set_ylabel("Classification accuracy", fontsize=16)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_ylim(0, 1.03)
        ax.legend(loc="lower right", fontsize=14)
        ax.grid()

        plots_path = (
            Path(cfg_dataset.paths.plots_path)
            / f"classify_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/"
        )
        plots_path.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(plots_path / "trainsize_vs_accuracy.png")


if __name__ == "__main__":
    main()
