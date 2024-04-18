import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from cca_zoo.linear import CCA
from omegaconf import DictConfig
from swarm_visualizer.histogram import (
    plot_several_pdf,
)
from swarm_visualizer.utility.general_utils import save_fig

from mmda.benchmark.asif_core import zero_shot_classification
from mmda.utils.data_utils import (
    load_CLIP_like_data,
    load_two_encoder_data,
    origin_centered,
)
from mmda.utils.dataset_utils import (
    get_train_test_split_index,
    shuffle_by_level,
    train_test_split,
)
from mmda.utils.sim_utils import (
    ROC_align_unalign_points,
    cosine_sim,
    weighted_corr_sim,
)


def CCA_data_align(cfg: DictConfig, shuffle_level: str = "dataset") -> list[tuple[float, float]]:
    """Align the audio and text data and calculate the similarity score using my proposed method.

    Args:
        cfg: configuration file
        shuffle_level: shuffle level. It can be "dataset", "class", or "object".

    Returns:
        ROC points

    """
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, Data1, Data2 = load_two_encoder_data(cfg)
    plots_path = os.path.join(cfg_dataset.paths.plots_path + "shuffle_align/")
    os.makedirs(plots_path, exist_ok=True)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Data1.shape[0])
    trainData1, valData1 = train_test_split(Data1, trainIdx, valIdx)
    trainData2, valData2 = train_test_split(Data2, trainIdx, valIdx)

    ### aligned case: not shuffle the data
    trainData1Align, valData1Align = trainData1.copy(), valData1.copy()
    trainData2Align, valData2Align = trainData2.copy(), valData2.copy()
    # zero mean data
    trainData1Align, trainData1_mean = origin_centered(trainData1Align)
    trainData2Align, trainData2_mean = origin_centered(trainData2Align)
    valData1Align = valData1Align - trainData1_mean
    valData2Align = valData2Align - trainData2_mean
    # make sure the data is zero mean
    assert np.allclose(
        trainData1Align.mean(axis=0), 0, atol=1e-4
    ), f"trainData1Align not zero mean: {trainData1Align.mean(axis=0)}"
    assert np.allclose(
        trainData2Align.mean(axis=0), 0, atol=1e-4
    ), f"trainData2Align not zero mean: {trainData2Align.mean(axis=0)}"

    # CCA dimensionality reduction
    cca = CCA(latent_dimensions=cfg_dataset.CCA_dim)
    trainData1Align, trainData2Align = cca.fit_transform((trainData1Align, trainData2Align))
    if cfg_dataset.equal_weights:
        corr_align = np.ones((trainData2Align.shape[1],))  # dim,
    else:
        corr_align = np.diag(trainData1Align.T @ trainData2Align) / trainData1Align.shape[0]  # dim,

    # calculate the similarity score
    valData1Align, valData2Align = cca.transform((valData1Align, valData2Align))
    sim_align = weighted_corr_sim(valData1Align, valData2Align, corr_align, dim=cfg_dataset.sim_dim)

    ### unaligned case: shuffle the data
    # shuffle only the text data
    trainData1Unalign, valData1Unalign = trainData1.copy(), valData1.copy()
    trainData2Unalign, valData2Unalign = trainData2.copy(), valData2.copy()

    trainData2Unalign, valData2Unalign = shuffle_by_level(
        cfg_dataset, cfg.dataset, shuffle_level, trainData2Unalign, valData2Unalign, trainIdx, valIdx
    )

    # zero mean data
    trainData1Unalign, trainData1_mean_ = origin_centered(trainData1Unalign)
    trainData2Unalign, trainData2_mean_ = origin_centered(trainData2Unalign)
    valData1Unalign = valData1Unalign - trainData1_mean_
    valData2Unalign = valData2Unalign - trainData2_mean_

    # make sure the data is zero mean
    assert np.allclose(
        trainData1Unalign.mean(axis=0), 0, atol=1e-4
    ), f"trainData1Unalign not zero mean: {trainData1Unalign.mean(axis=0)}"
    assert np.allclose(
        trainData2Unalign.mean(axis=0), 0, atol=1e-4
    ), f"trainData2Unalign not zero mean: {trainData2Unalign.mean(axis=0)}"

    valData1Align, valData2Align = cca.transform((valData1Unalign, valData2Unalign))
    sim_unalign = weighted_corr_sim(valData1Align, valData2Align, corr_align, dim=cfg_dataset.sim_dim)
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
        os.path.join(
            plots_path
            + f"similarity_score_{shuffle_level}_r{cfg.train_test_ratio}_dim{cfg_dataset.sim_dim}{eq_label}.png"
        ),
    )

    # CCA dimensionality reduction
    cca_unalign = CCA(latent_dimensions=cfg_dataset.CCA_dim)
    trainData1Unalign, trainData2Unalign = cca_unalign.fit_transform((trainData1Unalign, trainData2Unalign))
    corr_unalign = np.diag(trainData1Unalign.T @ trainData2Unalign) / trainData1Unalign.shape[0]

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
        fig.savefig(os.path.join(plots_path, "cca_corr.png"))

    # plot ROC
    ROC_points_list = ROC_align_unalign_points(sim_align, sim_unalign, (-0.15, 0.65, 40))
    return ROC_points_list


def CLIP_like_data_align(cfg: DictConfig, shuffle_level: str = "dataset") -> list[tuple[float, float]]:
    """Align the audio and text data and calculate the similarity score using CLIP like models.

    Args:
        cfg: configuration file
        shuffle_level: shuffle level. It can be "dataset", "class", or "object".

    Returns:
        ROC points
    """
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, Data1, Data2 = load_CLIP_like_data(cfg)
    clip_model_name = "CLAP" if cfg.dataset == "musiccaps" else "CLIP"
    plots_path = os.path.join(cfg_dataset.paths.plots_path + "shuffle_align/")
    os.makedirs(plots_path, exist_ok=True)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Data1.shape[0])
    _, valData1 = train_test_split(Data1, trainIdx, valIdx)
    trainData2, valData2 = train_test_split(Data2, trainIdx, valIdx)

    # copy data
    valData1Align = valData1.copy()
    valData2Align = valData2.copy()
    trainData2Unalign = trainData2.copy()
    valData2Unalign = valData2.copy()

    trainData2Unalign, valData2Unalign = shuffle_by_level(
        cfg_dataset, cfg.dataset, shuffle_level, trainData2Unalign, valData2Unalign, trainIdx, valIdx
    )

    sim_align = cosine_sim(valData1Align, valData2Align)
    sim_unalign = cosine_sim(valData1Align, valData2Unalign)

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
        os.path.join(plots_path, f"cos_similarity_{shuffle_level}_{clip_model_name}_r{cfg.train_test_ratio}.png"),
    )

    # plot ROC
    ROC_points_list = ROC_align_unalign_points(sim_align, sim_unalign, (-1, 1, 40))
    return ROC_points_list


def ASIF_data_align(cfg: DictConfig, shuffle_level: str = "dataset") -> list[tuple[float, float]]:
    """Align the audio and text data and calculate the similarity score using the ASIF method.

    Paper: https://openreview.net/pdf?id=YAxV_Krcdjm
    Args:
        cfg: configuration file
        shuffle_level: shuffle level. It can be "dataset", "class", or "object".

    Returns:
        ROC points
    """
    # load embeddings from the two encoders
    cfg_dataset, Data1, Data2 = load_two_encoder_data(cfg)
    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Data1.shape[0])
    trainData1, valData1 = train_test_split(Data1, trainIdx, valIdx)
    trainData2, valData2 = train_test_split(Data2, trainIdx, valIdx)

    # normalization to perform cosine similarity with a simple matmul
    trainData1 /= np.linalg.norm(trainData1, axis=1, keepdims=True)
    trainData2 /= np.linalg.norm(trainData2, axis=1, keepdims=True)
    valData1 /= np.linalg.norm(valData1, axis=1, keepdims=True)
    valData2 /= np.linalg.norm(valData2, axis=1, keepdims=True)

    # set parameters
    non_zeros = min(cfg.asif.non_zeros, trainData1.shape[0])
    range_anch = [2**i for i in range(int(np.log2(non_zeros) + 1), int(np.log2(len(trainData1))) + 2)]
    range_anch = range_anch[-1:]  # run just last anchor to be quick

    # copy data
    valData1Align = valData1.copy()
    valData2Align = valData2.copy()
    trainData2Unalign = trainData2.copy()
    valData2Unalign = valData2.copy()
    trainData2Unalign, valData2Unalign = shuffle_by_level(
        cfg_dataset, cfg.dataset, shuffle_level, trainData2Unalign, valData2Unalign, trainIdx, valIdx
    )

    # convert to torch tensors
    val_labels = torch.zeros(valData1.shape[0])  # dummy labels. Unused because this is not zero-shot classification.
    valData1Align, valData2Align = torch.tensor(valData1Align).cuda(), torch.tensor(valData2Align).cuda()
    valData2Unalign = torch.tensor(valData2Unalign).cuda()
    trainData1, trainData2 = torch.tensor(trainData1).cuda(), torch.tensor(trainData2).cuda()

    # similarity score of aligned data
    n_anchors, scores, sims = zero_shot_classification(
        valData1Align,
        valData2Align,
        trainData1,
        trainData2,
        val_labels,
        non_zeros,
        range_anch,
        cfg.asif.val_exps,
        max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
    )
    sim_align = np.diag(sims.detach().cpu().numpy())

    # similarity score of unaligned data
    n_anchors, scores, sims = zero_shot_classification(
        valData1Align,
        valData2Unalign,
        trainData1,
        trainData2,
        val_labels,
        non_zeros,
        range_anch,
        cfg.asif.val_exps,
        max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
    )
    sim_unalign = np.diag(sims.detach().cpu().numpy())

    # plot ROC
    ROC_points_list = ROC_align_unalign_points(sim_align, sim_unalign, (-1, 1, 40))
    return ROC_points_list

