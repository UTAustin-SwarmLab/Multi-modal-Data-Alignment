"""Train a linear SVM on the ImageNet dataset."""

# ruff: noqa: ERA001, PLR2004, S301

import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn import svm

import hydra
from mmda.utils.cca_class import NormalizedCCA
from mmda.utils.data_utils import load_two_encoder_data
from mmda.utils.dataset_utils import (
    get_train_test_split_index,
    load_imagenet,
    load_leafy_spurge,
    train_test_split,
)

BATCH_SIZE = 256


@hydra.main(version_base=None, config_path="../config", config_name="main")
def train_linear_svm(cfg: DictConfig) -> None:
    """Train a linear SVM on the ImageNet dataset."""
    np.random.seed(cfg.seed)
    cfg_dataset = cfg[cfg.dataset]
    if cfg.dataset == "imagenet":
        _, mturks_idx, labels, _ = load_imagenet(cfg_dataset)

        with Path(cfg_dataset.paths.save_path, "ImageNet_img_emb_clip.pkl").open(
            "rb"
        ) as f:
            img_emb = pickle.load(f)
    elif cfg.dataset == "leafy_spurge":
        _, labels, _ = load_leafy_spurge(cfg_dataset)
        with Path(cfg_dataset.paths.save_path, "LeafySpurge_img_emb_clip.pkl").open(
            "rb"
        ) as f:
            img_emb = pickle.load(f)

    # transform the data using CCA
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)

    # Train linear SVM
    start_time = time.time()
    train_idx, val_idx = get_train_test_split_index(
        cfg.train_test_ratio, img_emb.shape[0]
    )
    labels_train, labels_test = train_test_split(labels, train_idx, val_idx)
    print(labels_train.shape, labels_test.shape)

    # CSA case
    csa_train_data1, csa_val_data1 = train_test_split(data1, train_idx, val_idx)
    csa_train_data2, csa_val_data2 = train_test_split(data2, train_idx, val_idx)
    cca = NormalizedCCA()
    cca_img_train, cca_text_train, _ = cca.fit_transform_train_data(
        cfg_dataset, csa_train_data1, csa_train_data2
    )
    clf = svm.SVC(kernel="linear")
    clf.fit(cca_img_train, labels_train)
    cca_img_val, cca_text_val = cca.transform_data(csa_val_data1, csa_val_data2)
    y_pred = clf.predict(cca_img_val)
    accuracy = np.mean(y_pred == labels_test)
    print(f"CSA accuracy: {accuracy * 100:.2f}%")
    return

    # CLIP case
    x_train, x_test = train_test_split(img_emb, train_idx, val_idx)
    print(x_train.shape, x_test.shape)
    print(len(labels_train), len(labels_test))
    clf = svm.SVC(kernel="linear")
    clf.fit(x_train, labels_train)

    end_time = time.time()

    print(f"Training time: {end_time - start_time:.2f} seconds")
    y_pred = clf.predict(x_test)
    accuracy = np.mean(y_pred == labels_test)
    print(f"Split {cfg.train_test_ratio} accuracy: {accuracy * 100:.2f}%")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def plot_accuracy(cfg: DictConfig) -> None:
    """Plot the accuracy of the model."""
    cfg_dataset = cfg[cfg.dataset]
    ds_size = 50_000 if cfg.dataset == "imagenet" else 900
    csv_save_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"classify_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/"
        / f"accuracy_{cfg_dataset.sim_dim}_svm.csv"
    )
    df = pd.read_csv(csv_save_path)
    ratios = df["train_test_ratio"] * ds_size
    cca_accs = df["cca_accs"]
    asif_accs = df["asif_accs"]
    fig, ax = plt.subplots()
    ax.plot(
        ratios,
        cca_accs,
        "o-",
        ms=12,
        label="CSA (ours)",
        color="blue",
    )
    clip_accs = df["clip_accs"]
    ax.plot(
        ratios,
        clip_accs,
        "^--",
        ms=12,
        label="CLIP",
        color="red",
    )
    clip_svm_accs = df["svm_accs"]
    ax.plot(
        ratios,
        clip_svm_accs,
        "v--",
        ms=12,
        label="CLIP + Linear SVM",
        color="orange",
    )
    csa_svm_accs = df["csa_svm_accs"]
    ax.plot(
        ratios,
        csa_svm_accs,
        "D-.",
        ms=12,
        label="CSA + Linear SVM",
        color="purple",
    )
    ax.plot(
        ratios,
        asif_accs,
        "D-.",
        ms=12,
        label="ASIF",
        color="green",
    )
    ax.set_xlabel("Amount of training data", fontsize=20)
    ax.set_ylabel("Classification accuracy", fontsize=20)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_ylim(-0.1, 1.1) if cfg.dataset == "imagenet" else ax.set_ylim(0.3, 0.7)
    ax.legend(loc="lower right", fontsize=16)
    ax.grid()

    plots_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"classify_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/"
    )
    plots_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(plots_path / f"trainsize_vs_accuracy_svm{cfg_dataset.sim_dim}.png")


if __name__ == "__main__":
    # train_linear_svm()
    plot_accuracy()

# CUDA_VISIBLE_DEVICES=5 poetry run python mmda/linear_svm_clip.py dataset=leafy_spurge leafy_spurge.sim_dim=250
