"""This module contains the functions to detect shuffled text descriptions using the proposed method and baselines."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from swarm_visualizer.histogram import plot_several_pdf
from swarm_visualizer.utility.general_utils import save_fig

from mmda.baselines.asif_core import zero_shot_classification
from mmda.utils.cca import cca_fit_train_data
from mmda.utils.data_utils import (
    load_clip_like_data,
    load_two_encoder_data,
    origin_centered,
)
from mmda.utils.dataset_utils import (
    get_train_test_split_index,
    shuffle_by_level,
    train_test_split,
)
from mmda.utils.sim_utils import (
    cosine_sim,
    roc_align_unalign_points,
    weighted_corr_sim,
)


def cca_data_align(
    cfg: DictConfig, shuffle_level: str = "dataset"
) -> list[tuple[float, float]]:
    """Align the audio and text data and calculate the similarity score using my proposed method.

    Args:
        cfg: configuration file
        shuffle_level: shuffle level. It can be "dataset", "class", or "object".

    Returns:
        ROC points

    """
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    plots_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"shuffle_align_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/"
    )
    plots_path.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx = get_train_test_split_index(
        cfg.train_test_ratio, data1.shape[0]
    )
    traindata1, valdata1 = train_test_split(data1, train_idx, val_idx)
    traindata2, valdata2 = train_test_split(data2, train_idx, val_idx)

    ### aligned case: not shuffle the data
    traindata1align, valdata1align = traindata1.copy(), valdata1.copy()
    traindata2align, valdata2align = traindata2.copy(), valdata2.copy()
    # zero mean data
    traindata1align, traindata1_mean = origin_centered(traindata1align)
    traindata2align, traindata2_mean = origin_centered(traindata2align)
    valdata1align = valdata1align - traindata1_mean
    valdata2align = valdata2align - traindata2_mean
    # make sure the data is zero mean
    assert np.allclose(
        traindata1align.mean(axis=0), 0, atol=1e-4
    ), f"traindata1align not zero mean: {traindata1align.mean(axis=0)}"
    assert np.allclose(
        traindata2align.mean(axis=0), 0, atol=1e-4
    ), f"traindata2align not zero mean: {traindata2align.mean(axis=0)}"

    cca, traindata1align, traindata2align, corr_align = cca_fit_train_data(
        cfg_dataset, traindata1align, traindata2align
    )

    # calculate the similarity score
    valdata1align, valdata2align = cca.transform((valdata1align, valdata2align))
    sim_align = weighted_corr_sim(
        valdata1align, valdata2align, corr_align, dim=cfg_dataset.sim_dim
    )

    ### unaligned case: shuffle the data
    # shuffle only the text data
    traindata1unalign, valdata1unalign = traindata1.copy(), valdata1.copy()
    traindata2unalign, valdata2unalign = traindata2.copy(), valdata2.copy()

    traindata2unalign, valdata2unalign = shuffle_by_level(
        cfg_dataset,
        cfg.dataset,
        shuffle_level,
        traindata2unalign,
        valdata2unalign,
        train_idx,
        val_idx,
    )

    # zero mean data
    traindata1unalign, traindata1_mean_ = origin_centered(traindata1unalign)
    traindata2unalign, traindata2_mean_ = origin_centered(traindata2unalign)
    valdata1unalign = valdata1unalign - traindata1_mean_
    valdata2unalign = valdata2unalign - traindata2_mean_

    # make sure the data is zero mean
    assert np.allclose(
        traindata1unalign.mean(axis=0), 0, atol=1e-4
    ), f"traindata1unalign not zero mean: {traindata1unalign.mean(axis=0)}"
    assert np.allclose(
        traindata2unalign.mean(axis=0), 0, atol=1e-4
    ), f"traindata2unalign not zero mean: {traindata2unalign.mean(axis=0)}"

    valdata1align, valdata2align = cca.transform((valdata1unalign, valdata2unalign))
    sim_unalign = weighted_corr_sim(
        valdata1align, valdata2align, corr_align, dim=cfg_dataset.sim_dim
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(
        data_list=[sim_align, sim_unalign],
        legend=["Aligned", "Unaligned"],
        title_str="Similarity Score Distribution",
        xlabel="Similarity Score",
        ylabel="Frequency",
        ax=ax,
    )
    eq_label = "_noweight" if cfg_dataset.equal_weights else ""
    save_fig(
        fig,
        plots_path
        / f"similarity_score_{shuffle_level}_r{cfg.train_test_ratio}_dim{cfg_dataset.sim_dim}{eq_label}.png",
    )

    cca_unalign, traindata1unalign, traindata2unalign, corr_unalign = (
        cca_fit_train_data(cfg_dataset, traindata1unalign, traindata2unalign)
    )

    # plot the correlation coefficients
    if not cfg_dataset.equal_weights:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(corr_align)
        ax.plot(corr_unalign)
        ax.set_title("Correlation Coefficients of the Cross Covariance")
        ax.set_xlabel("Dimension of Eigenvalues")
        ax.set_ylabel("Correlation Coefficients")
        ax.legend(["Aligned", "Unaligned"])
        ax.set_ylim(0, 1)
        fig.savefig(plots_path / "cca_corr.png")

    # plot ROC
    return roc_align_unalign_points(sim_align, sim_unalign, (-1.0, 1.0, 80))


def clip_like_data_align(
    cfg: DictConfig, shuffle_level: str = "dataset"
) -> list[tuple[float, float]]:
    """Align the audio and text data and calculate the similarity score using CLIP like models.

    Args:
        cfg: configuration file
        shuffle_level: shuffle level. It can be "dataset", "class", or "object".

    Returns:
        ROC points
    """
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, data1, data2 = load_clip_like_data(cfg)
    clip_model_name = "CLAP" if cfg.dataset == "musiccaps" else "CLIP"
    plots_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"shuffle_align_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/"
    )
    plots_path.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx = get_train_test_split_index(
        cfg.train_test_ratio, data1.shape[0]
    )
    _, valdata1 = train_test_split(data1, train_idx, val_idx)
    traindata2, valdata2 = train_test_split(data2, train_idx, val_idx)

    # copy data
    valdata1align = valdata1.copy()
    valdata2align = valdata2.copy()
    traindata2unalign = traindata2.copy()
    valdata2unalign = valdata2.copy()

    traindata2unalign, valdata2unalign = shuffle_by_level(
        cfg_dataset,
        cfg.dataset,
        shuffle_level,
        traindata2unalign,
        valdata2unalign,
        train_idx,
        val_idx,
    )

    sim_align = cosine_sim(valdata1align, valdata2align)
    sim_unalign = cosine_sim(valdata1align, valdata2unalign)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(
        data_list=[sim_align, sim_unalign],
        legend=["Aligned", "Random shuffle"],
        title_str="Similarity Score Distribution",
        xlabel="Similarity Score",
        ylabel="Frequency",
        ax=ax,
    )
    save_fig(
        fig,
        plots_path
        / f"cos_similarity_{shuffle_level}_{clip_model_name}_r{cfg.train_test_ratio}.png",
    )

    # plot ROC
    return roc_align_unalign_points(sim_align, sim_unalign, (-1, 1, 80))


def asif_data_align(
    cfg: DictConfig, shuffle_level: str = "dataset"
) -> list[tuple[float, float]]:
    """Align the audio and text data and calculate the similarity score using the ASIF method.

    Paper: https://openreview.net/pdf?id=YAxV_Krcdjm
    Args:
        cfg: configuration file
        shuffle_level: shuffle level. It can be "dataset", "class", or "object".

    Returns:
        ROC points
    """
    # load embeddings from the two encoders
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    train_idx, val_idx = get_train_test_split_index(
        cfg.train_test_ratio, data1.shape[0]
    )
    traindata1, valdata1 = train_test_split(data1, train_idx, val_idx)
    traindata2, valdata2 = train_test_split(data2, train_idx, val_idx)

    # normalization to perform cosine similarity with a simple matmul
    traindata1 /= np.linalg.norm(traindata1, axis=1, keepdims=True)
    traindata2 /= np.linalg.norm(traindata2, axis=1, keepdims=True)
    valdata1 /= np.linalg.norm(valdata1, axis=1, keepdims=True)
    valdata2 /= np.linalg.norm(valdata2, axis=1, keepdims=True)

    # set parameters
    non_zeros = min(cfg.asif.non_zeros, traindata1.shape[0])
    range_anch = [
        2**i
        for i in range(int(np.log2(non_zeros) + 1), int(np.log2(len(traindata1))) + 2)
    ]
    range_anch = range_anch[-1:]  # run just last anchor to be quick

    # copy data
    valdata1align = valdata1.copy()
    valdata2align = valdata2.copy()
    traindata2unalign = traindata2.copy()
    valdata2unalign = valdata2.copy()
    traindata2unalign, valdata2unalign = shuffle_by_level(
        cfg_dataset,
        cfg.dataset,
        shuffle_level,
        traindata2unalign,
        valdata2unalign,
        train_idx,
        val_idx,
    )

    # convert to torch tensors
    val_labels = torch.zeros(
        valdata1.shape[0]
    )  # dummy labels. Unused because this is not zero-shot classification.
    valdata1align, valdata2align = (
        torch.tensor(valdata1align).cuda(),
        torch.tensor(valdata2align).cuda(),
    )
    valdata2unalign = torch.tensor(valdata2unalign).cuda()
    traindata1, traindata2 = (
        torch.tensor(traindata1).cuda(),
        torch.tensor(traindata2).cuda(),
    )

    # similarity score of aligned data
    n_anchors, scores, sims = zero_shot_classification(
        valdata1align,
        valdata2align,
        traindata1,
        traindata2,
        val_labels,
        non_zeros,
        range_anch,
        cfg.asif.val_exps,
        max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
    )
    sim_align = np.diag(sims.detach().cpu().numpy())

    # similarity score of unaligned data
    n_anchors, scores, sims = zero_shot_classification(
        valdata1align,
        valdata2unalign,
        traindata1,
        traindata2,
        val_labels,
        non_zeros,
        range_anch,
        cfg.asif.val_exps,
        max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
    )
    sim_unalign = np.diag(sims.detach().cpu().numpy())

    # plot ROC
    return roc_align_unalign_points(sim_align, sim_unalign, (-1, 1, 150))
