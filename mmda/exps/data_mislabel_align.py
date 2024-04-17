"""This module contains the functions to detect mislabeled data using the proposed method and baselines."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from swarm_visualizer.histogram import (
    plot_several_pdf,
)
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
    load_cosmos,
    load_imagenet,
    load_pitts,
    load_tiil,
    train_test_split,
)
from mmda.utils.sim_utils import (
    cosine_sim,
    roc_align_unalign_points,
    weighted_corr_sim,
)


def cca_detect_mislabeled_data(cfg: DictConfig) -> list[tuple[float, float]]:
    """Detect mislabeled data using the proposed CCA method.

    Args:
        cfg: configuration file

    Returns:
        ROC points

    """
    np.random.seed(cfg.seed)
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    plots_path = Path(cfg_dataset.paths.plots_path) / "mislabeled/"
    plots_path.mkdir(parents=True, exist_ok=True)

    wrong_labels_bool = parse_wrong_label(cfg)

    train_idx, val_idx = get_train_test_split_index(cfg.train_test_ratio, data1.shape[0])
    traindata1, valdata1 = train_test_split(data1, train_idx, val_idx)
    traindata2, valdata2 = train_test_split(data2, train_idx, val_idx)
    train_wrong_labels_bool, val_wrong_labels_bool = train_test_split(wrong_labels_bool, train_idx, val_idx)

    # separate aligned data and unaligned data
    traindata1align, traindata2align = (
        traindata1[~train_wrong_labels_bool],
        traindata2[~train_wrong_labels_bool],
    )
    valdata1align, valdata2align = (
        valdata1[~val_wrong_labels_bool],
        valdata2[~val_wrong_labels_bool],
    )
    valdata1unalign, valdata2unalign = (
        valdata1[val_wrong_labels_bool],
        valdata2[val_wrong_labels_bool],
    )

    # select training data based on the noisy_train_set
    traindata1 = traindata1 if cfg.noisy_train_set else traindata1align
    traindata2 = traindata2 if cfg.noisy_train_set else traindata2align
    train_label = "" if cfg.noisy_train_set else "_clean"
    eq_label = "_noweight" if cfg_dataset.equal_weights else ""

    # zero mean data
    traindata1, traindata1_mean = origin_centered(traindata1)
    traindata2, traindata2_mean = origin_centered(traindata2)
    valdata1align = valdata1align - traindata1_mean
    valdata2align = valdata2align - traindata2_mean
    # make sure the data is zero mean
    assert np.allclose(
        traindata1.mean(axis=0), 0, atol=1e-4
    ), f"traindata1align not zero mean: {traindata1.mean(axis=0)}"
    assert np.allclose(
        traindata2.mean(axis=0), 0, atol=1e-4
    ), f"traindata2align not zero mean: {traindata2.mean(axis=0)}"

    cca, traindata1, traindata2, corr_align = cca_fit_train_data(cfg_dataset, traindata1, traindata2)

    # calculate the similarity score
    valdata1align, valdata2align = cca.transform((valdata1align, valdata2align))
    sim_align = weighted_corr_sim(valdata1align, valdata2align, corr_align, dim=cfg_dataset.sim_dim)

    ### unaligned case: shuffle the data
    # zero mean data
    valdata1unalign = valdata1unalign - traindata1_mean
    valdata2unalign = valdata2unalign - traindata2_mean

    valdata1unalign, valdata2unalign = cca.transform((valdata1unalign, valdata2unalign))
    sim_unalign = weighted_corr_sim(valdata1unalign, valdata2unalign, corr_align, dim=cfg_dataset.sim_dim)
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
        plots_path
        / f"similarity_score_mislabeled_r{cfg.train_test_ratio}_dim{cfg_dataset.sim_dim}{eq_label}{train_label}.png",
    )

    # plot ROC
    return roc_align_unalign_points(sim_align, sim_unalign, (-1.0, 1.0, 80))


def clip_like_detect_mislabeled_data(cfg: DictConfig) -> list[tuple[float, float]]:
    """Detect mislabeled data using the CLIP-like models.

    Args:
        cfg: configuration file

    Returns:
        ROC points
    """
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, data1, data2 = load_clip_like_data(cfg)
    clip_model_name = "CLAP" if cfg.dataset == "musiccaps" else "CLIP"
    plots_path = Path(cfg_dataset.paths.plots_path) / "mislabeled/"
    plots_path.mkdir(parents=True, exist_ok=True)

    wrong_labels_bool = parse_wrong_label(cfg)

    train_idx, val_idx = get_train_test_split_index(cfg.train_test_ratio, data1.shape[0])
    traindata1, valdata1 = train_test_split(data1, train_idx, val_idx)
    traindata2, valdata2 = train_test_split(data2, train_idx, val_idx)
    train_wrong_labels_bool, val_wrong_labels_bool = train_test_split(wrong_labels_bool, train_idx, val_idx)

    # separate aligned data and unaligned data
    valdata1align, valdata2align = (
        valdata1[~val_wrong_labels_bool],
        valdata2[~val_wrong_labels_bool],
    )
    valdata1unalign, valdata2unalign = (
        valdata1[val_wrong_labels_bool],
        valdata2[val_wrong_labels_bool],
    )

    sim_align = cosine_sim(valdata1align, valdata2align)
    sim_unalign = cosine_sim(valdata1unalign, valdata2unalign)

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
        plots_path / f"cos_similarity_mislabeled_{clip_model_name}_r{cfg.train_test_ratio}.png",
    )

    # plot ROC
    return roc_align_unalign_points(sim_align, sim_unalign, (-1, 1, 50))


def asif_detect_mislabeled_data(cfg: DictConfig) -> list[tuple[float, float]]:
    """Detect mislabeled data using the ASIF method.

    Paper: https://openreview.net/pdf?id=YAxV_Krcdjm
    Args:
        cfg: configuration file

    Returns:
        ROC points
    """
    wrong_labels_bool = parse_wrong_label(cfg)

    # load embeddings from the two encoders
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    train_idx, val_idx = get_train_test_split_index(cfg.train_test_ratio, data1.shape[0])
    traindata1, valdata1 = train_test_split(data1, train_idx, val_idx)
    traindata2, valdata2 = train_test_split(data2, train_idx, val_idx)
    train_wrong_labels_bool, val_wrong_labels_bool = train_test_split(wrong_labels_bool, train_idx, val_idx)

    # normalization to perform cosine similarity with a simple matmul
    traindata1 /= np.linalg.norm(traindata1, axis=1, keepdims=True)
    traindata2 /= np.linalg.norm(traindata2, axis=1, keepdims=True)
    valdata1 /= np.linalg.norm(valdata1, axis=1, keepdims=True)
    valdata2 /= np.linalg.norm(valdata2, axis=1, keepdims=True)

    # set parameters
    non_zeros = min(cfg.asif.non_zeros, traindata1.shape[0])
    range_anch = [2**i for i in range(int(np.log2(non_zeros) + 1), int(np.log2(len(traindata1))) + 2)]
    range_anch = range_anch[-1:]  # run just last anchor to be quick

    # convert to torch tensors
    wrong_labels_bool = torch.tensor(wrong_labels_bool).cuda()
    valdata1, valdata2 = torch.tensor(valdata1).cuda(), torch.tensor(valdata2).cuda()
    traindata1, traindata2 = (
        torch.tensor(traindata1).cuda(),
        torch.tensor(traindata2).cuda(),
    )

    # separate aligned data and unaligned data
    traindata1align, traindata2align = (
        traindata1[~train_wrong_labels_bool],
        traindata2[~train_wrong_labels_bool],
    )
    valdata1align, valdata2align = (
        valdata1[~val_wrong_labels_bool],
        valdata2[~val_wrong_labels_bool],
    )
    valdata1unalign, valdata2unalign = (
        valdata1[val_wrong_labels_bool],
        valdata2[val_wrong_labels_bool],
    )

    # similarity score of val data
    n_anchors, scores, sims = zero_shot_classification(
        valdata1align,
        valdata2align,
        traindata1 if cfg.noisy_train_set else traindata1align,
        traindata2 if cfg.noisy_train_set else traindata2align,
        torch.zeros(valdata1align.shape[0]),
        non_zeros,
        range_anch,
        cfg.asif.val_exps,
        max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
    )
    sim_align = np.diag(sims.detach().cpu().numpy())

    # similarity score of unaligned val data
    n_anchors, scores, sims = zero_shot_classification(
        valdata1unalign,
        valdata2unalign,
        traindata1 if cfg.noisy_train_set else traindata1align,
        traindata2 if cfg.noisy_train_set else traindata2align,
        torch.zeros(valdata1unalign.shape[0]),
        non_zeros,
        range_anch,
        cfg.asif.val_exps,
        max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
    )

    sim_unalign = np.diag(sims.detach().cpu().numpy())

    # plot ROC
    return roc_align_unalign_points(sim_align, sim_unalign, (-1, 1, 50))


def parse_wrong_label(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """Parse the wrong label boolean array to the train and val set.

    Args:
        cfg: configuration file
        wrong_labels_bool: boolean array indicating whether the label is wrong.
        train_idx: index of the training set
        val_idx: index of the validation set

    Returns:
        wrong_labels_bool
    """
    if cfg.dataset == "imagenet":
        cfg_dataset = cfg.imagenet
        img_path, mturks_idx, orig_idx, clsidx_to_labels = load_imagenet(cfg_dataset)
        wrong_labels_bool = []  # True if the label is wrong, False if the label is correct
        for mturks_label_idx, orig_label_idx in zip(mturks_idx, orig_idx):
            (wrong_labels_bool.append(True) if mturks_label_idx != orig_label_idx else wrong_labels_bool.append(False))
        wrong_labels_bool = np.array(wrong_labels_bool, dtype=bool)
    elif cfg.dataset == "tiil":
        cfg_dataset = cfg.tiil
        img_paths, text_desciption, wrong_labels_bool, _ = load_tiil(cfg_dataset)
    elif cfg.dataset == "cosmos":
        cfg_dataset = cfg.cosmos
        img_paths, text_desciption, wrong_labels_bool, _ = load_cosmos(cfg_dataset)
    elif cfg.dataset == "pitts":
        cfg_dataset = cfg.pitts
        img_paths, text_desciption, wrong_labels_bool = load_pitts(cfg_dataset)
    # TODO: add more datasets
    else:
        raise NotImplementedError
    return wrong_labels_bool
