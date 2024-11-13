"""Plot the T-SNE of the CSA embeddings on ImageNet."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.manifold import TSNE

import hydra
from mmda.exps.mislabel_align import separate_data
from mmda.utils.cca_class import NormalizedCCA
from mmda.utils.data_utils import load_two_encoder_data
from mmda.utils.dataset_utils import load_imagenet

cell_size = 30
label_size = 30
ticks_size = 26


def plot_imagenet_tsne(cfg: DictConfig, save: bool = False) -> None:
    """Plot the T-SNE of the CSA embeddings on ImageNet."""
    ### load embeddings ###
    img_path, mturks_idx, orig_idx, clsidx_to_labels = load_imagenet(cfg.imagenet)

    np.random.seed(cfg.seed)
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    print(f"Loaded data1 shape: {data1.shape}, data2 shape: {data2.shape}")
    alldata = separate_data(cfg, data1, data2)

    # we only consider the correctly labeled data

    # select training data based on the noisy_train_set
    traindata1 = alldata.traindata1align
    traindata2 = alldata.traindata2align
    train_idx = alldata.train_idx[~alldata.train_wrong_labels_bool]
    print(
        f"img_data shape: {traindata1.shape}, mturks_idx[train_idx] shape: {mturks_idx[train_idx].shape}"
    )
    class_20_idx = mturks_idx[train_idx] % 50 == 0
    print(f"val_idx shape: {class_20_idx.shape}")

    # transform the data using CCA
    cca = NormalizedCCA()
    cca_img_data, cca_text_data, _ = cca.fit_transform_train_data(
        cfg_dataset, traindata1, traindata2
    )
    print(f"cca_img_data shape: {cca_img_data.shape}")

    # Compute t-SNE for original image embeddings
    tsne_img = TSNE(n_components=2, random_state=cfg.seed).fit_transform(
        traindata1[class_20_idx]
    )

    # Compute t-SNE for CCA-transformed embeddings
    tsne_cca = TSNE(n_components=2, random_state=cfg.seed).fit_transform(
        cca_img_data[class_20_idx]
    )
    # Plot original image embeddings
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    print(f"idx: {mturks_idx[train_idx][class_20_idx].shape}")
    print(f"embeddings: {tsne_img.shape}")
    _ = ax1.scatter(
        tsne_img[:, 0],
        tsne_img[:, 1],
        c=mturks_idx[train_idx][class_20_idx],
        cmap="tab20",
        alpha=0.8,
    )
    ax1.set_xlabel("t-SNE dimension 1", fontsize=label_size)
    ax1.set_ylabel("t-SNE dimension 2", fontsize=label_size)
    ax1.tick_params(axis="both", labelsize=ticks_size)
    plt.tight_layout()

    # Plot CCA-transformed embeddings
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    _ = ax2.scatter(
        tsne_cca[:, 0],
        tsne_cca[:, 1],
        c=mturks_idx[train_idx][class_20_idx],
        cmap="tab20",
        alpha=0.8,
    )
    ax2.set_xlabel("t-SNE dimension 1", fontsize=label_size)
    ax2.set_ylabel("t-SNE dimension 2", fontsize=label_size)
    ax2.tick_params(axis="both", labelsize=ticks_size)
    plt.tight_layout()

    # Save plots if specified
    if save:
        plots_path = Path(
            cfg_dataset.paths.plots_path,
            f"tsne_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/",
        )
        plots_path.mkdir(parents=True, exist_ok=True)
        fig1.savefig(plots_path / "tsne_clip.png")
        fig2.savefig(plots_path / "tsne_csa.png")
        plt.close("all")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: D103
    plot_imagenet_tsne(cfg, save=True)


if __name__ == "__main__":
    main()
