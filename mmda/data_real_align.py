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
    load_ImageNet,
    train_test_split,
)
from mmda.utils.sim_utils import (
    ROC_align_unalign_points,
    cosine_sim,
    weighted_corr_sim,
)


def CCA_detect_mislabeled_data(cfg: DictConfig) -> list[tuple[float, float]]:
    """Detect mislabeled data using the proposed CCA method.

    Args:
        cfg: configuration file

    Returns:
        ROC points

    """
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, Data1, Data2 = load_two_encoder_data(cfg)
    plots_path = os.path.join(cfg_dataset.paths.plots_path + "mislabeled/")
    os.mkdir(plots_path) if not os.path.exists(plots_path) else None

    wrong_labels_bool = parse_wrong_label(cfg)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Data1.shape[0])
    trainData1, valData1 = train_test_split(Data1, trainIdx, valIdx)
    trainData2, valData2 = train_test_split(Data2, trainIdx, valIdx)
    train_wrong_labels_bool, val_wrong_labels_bool = train_test_split(wrong_labels_bool, trainIdx, valIdx)

    # separate aligned data and unaligned data
    trainData1Align, trainData2Align = (
        trainData1[~train_wrong_labels_bool],
        trainData2[~train_wrong_labels_bool],
    )
    valData1Align, valData2Align = (
        valData1[~val_wrong_labels_bool],
        valData2[~val_wrong_labels_bool],
    )
    valData1Unalign, valData2Unalign = valData1[val_wrong_labels_bool], valData2[val_wrong_labels_bool]

    # select training data based on the noisy_train_set
    trainData1 = trainData1 if cfg.noisy_train_set else trainData1Align
    trainData2 = trainData2 if cfg.noisy_train_set else trainData2Align
    train_label = "" if cfg.noisy_train_set else "_clean"
    eq_label = "_noweight" if cfg_dataset.equal_weights else ""

    # zero mean data
    trainData1, trainData1_mean = origin_centered(trainData1)
    trainData2, trainData2_mean = origin_centered(trainData2)
    valData1Align = valData1Align - trainData1_mean
    valData2Align = valData2Align - trainData2_mean
    # make sure the data is zero mean
    assert np.allclose(
        trainData1.mean(axis=0), 0, atol=1e-4
    ), f"trainData1Align not zero mean: {trainData1.mean(axis=0)}"
    assert np.allclose(
        trainData2.mean(axis=0), 0, atol=1e-4
    ), f"trainData2Align not zero mean: {trainData2.mean(axis=0)}"

    # CCA dimensionality reduction
    cca = CCA(latent_dimensions=cfg_dataset.CCA_dim)
    trainData1, trainData2 = cca.fit_transform((trainData1, trainData2))
    if cfg_dataset.equal_weights:
        corr_align = np.ones((trainData2.shape[1],))  # dim,
    else:
        corr_align = np.diag(trainData1.T @ trainData2) / trainData1.shape[0]  # dim,

    # calculate the similarity score
    valData1Align, valData2Align = cca.transform((valData1Align, valData2Align))
    sim_align = weighted_corr_sim(valData1Align, valData2Align, corr_align, dim=cfg_dataset.sim_dim)

    ### unaligned case: shuffle the data
    # zero mean data
    valData1Unalign = valData1Unalign - trainData1_mean
    valData2Unalign = valData2Unalign - trainData2_mean

    valData1Unalign, valData2Unalign = cca.transform((valData1Unalign, valData2Unalign))
    sim_unalign = weighted_corr_sim(valData1Unalign, valData2Unalign, corr_align, dim=cfg_dataset.sim_dim)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(
        data_list=[sim_align, sim_unalign],
        legend=["Aligned", "Unaligned"],
        title_str="Similarity Score Distribution",
        xlabel="Similarity Score",
        ylabel="Frequency",
        ax=ax,
    )

    save_fig(
        fig,
        os.path.join(
            plots_path,
            f"similarity_score_mislabeled_r{cfg.train_test_ratio}_dim{cfg_dataset.sim_dim}{eq_label}{train_label}.png",
        ),
    )

    # plot ROC
    ROC_points_list = ROC_align_unalign_points(sim_align, sim_unalign, (-0.25, 1, 50))
    return ROC_points_list


def CLIP_like_detect_mislabeled_data(cfg: DictConfig) -> list[tuple[float, float]]:
    """Detect mislabeled data using the CLIP-like models.

    Args:
        cfg: configuration file

    Returns:
        ROC points
    """
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, Data1, Data2 = load_CLIP_like_data(cfg)
    clip_model_name = "CLAP" if cfg.dataset == "musiccaps" else "CLIP"
    plots_path = os.path.join(cfg_dataset.paths.plots_path + "mislabeled/")

    wrong_labels_bool = parse_wrong_label(cfg)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Data1.shape[0])
    trainData1, valData1 = train_test_split(Data1, trainIdx, valIdx)
    trainData2, valData2 = train_test_split(Data2, trainIdx, valIdx)
    train_wrong_labels_bool, val_wrong_labels_bool = train_test_split(wrong_labels_bool, trainIdx, valIdx)

    # separate aligned data and unaligned data
    valData1Align, valData2Align = (
        valData1[~val_wrong_labels_bool],
        valData2[~val_wrong_labels_bool],
    )
    valData1Unalign, valData2Unalign = valData1[val_wrong_labels_bool], valData2[val_wrong_labels_bool]

    sim_align = cosine_sim(valData1Align, valData2Align)
    sim_unalign = cosine_sim(valData1Unalign, valData2Unalign)

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
        os.path.join(plots_path, f"cos_similarity_mislabeled_{clip_model_name}_r{cfg.train_test_ratio}.png"),
    )

    # plot ROC
    ROC_points_list = ROC_align_unalign_points(sim_align, sim_unalign, (-1, 1, 50))
    return ROC_points_list


def ASIF_detect_mislabeled_data(cfg: DictConfig) -> list[tuple[float, float]]:
    """Detect mislabeled data using the ASIF method.

    Paper: https://openreview.net/pdf?id=YAxV_Krcdjm
    Args:
        cfg: configuration file

    Returns:
        ROC points
    """
    wrong_labels_bool = parse_wrong_label(cfg)

    # load embeddings from the two encoders
    cfg_dataset, Data1, Data2 = load_two_encoder_data(cfg)
    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Data1.shape[0])
    trainData1, valData1 = train_test_split(Data1, trainIdx, valIdx)
    trainData2, valData2 = train_test_split(Data2, trainIdx, valIdx)
    train_wrong_labels_bool, val_wrong_labels_bool = train_test_split(wrong_labels_bool, trainIdx, valIdx)

    # normalization to perform cosine similarity with a simple matmul
    trainData1 /= np.linalg.norm(trainData1, axis=1, keepdims=True)
    trainData2 /= np.linalg.norm(trainData2, axis=1, keepdims=True)
    valData1 /= np.linalg.norm(valData1, axis=1, keepdims=True)
    valData2 /= np.linalg.norm(valData2, axis=1, keepdims=True)

    # set parameters
    non_zeros = min(cfg.asif.non_zeros, trainData1.shape[0])
    range_anch = [2**i for i in range(int(np.log2(non_zeros) + 1), int(np.log2(len(trainData1))) + 2)]
    range_anch = range_anch[-1:]  # run just last anchor to be quick

    # convert to torch tensors
    wrong_labels_bool = torch.tensor(wrong_labels_bool).cuda()
    valData1, valData2 = torch.tensor(valData1).cuda(), torch.tensor(valData2).cuda()
    trainData1, trainData2 = torch.tensor(trainData1).cuda(), torch.tensor(trainData2).cuda()

    # separate aligned data and unaligned data
    trainData1Align, trainData2Align = (
        trainData1[~train_wrong_labels_bool],
        trainData2[~train_wrong_labels_bool],
    )
    valData1Align, valData2Align = (
        valData1[~val_wrong_labels_bool],
        valData2[~val_wrong_labels_bool],
    )
    valData1Unalign, valData2Unalign = valData1[val_wrong_labels_bool], valData2[val_wrong_labels_bool]

    # similarity score of val data
    n_anchors, scores, sims = zero_shot_classification(
        valData1Align,
        valData2Align,
        trainData1 if cfg.noisy_train_set else trainData1Align,
        trainData2 if cfg.noisy_train_set else trainData2Align,
        torch.zeros(valData1Align.shape[0]),
        non_zeros,
        range_anch,
        cfg.asif.val_exps,
        max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
    )
    sim_align = np.diag(sims.detach().cpu().numpy())

    # similarity score of unaligned val data
    n_anchors, scores, sims = zero_shot_classification(
        valData1Unalign,
        valData2Unalign,
        trainData1 if cfg.noisy_train_set else trainData1Align,
        trainData2 if cfg.noisy_train_set else trainData2Align,
        torch.zeros(valData1Unalign.shape[0]),
        non_zeros,
        range_anch,
        cfg.asif.val_exps,
        max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
    )

    sim_unalign = np.diag(sims.detach().cpu().numpy())

    # plot ROC
    ROC_points_list = ROC_align_unalign_points(sim_align, sim_unalign, (-1, 1, 50))
    return ROC_points_list


def parse_wrong_label(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """Parse the wrong label boolean array to the train and val set.

    Args:
        cfg: configuration file
        wrong_labels_bool: boolean array indicating whether the label is wrong.
        trainIdx: index of the training set
        valIdx: index of the validation set

    Returns:
        wrong_labels_bool
    """
    if cfg.dataset == "imagenet":
        cfg_dataset = cfg.imagenet
        img_path, mturks_idx, orig_idx, clsidx_to_labels = load_ImageNet(cfg_dataset)
        wrong_labels_bool = []  # True if the label is wrong, False if the label is correct
        for mturks_label_idx, orig_label_idx in zip(mturks_idx, orig_idx):
            wrong_labels_bool.append(True) if mturks_label_idx != orig_label_idx else wrong_labels_bool.append(False)
        wrong_labels_bool = np.array(wrong_labels_bool, dtype=bool)
    elif cfg.dataset == "tiil":
        raise NotImplementedError
    return wrong_labels_bool
