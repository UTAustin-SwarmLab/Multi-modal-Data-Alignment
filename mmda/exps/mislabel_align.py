"""This module contains the functions to detect mislabeled data using the proposed method and baselines."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from swarm_visualizer.histogram import plot_several_pdf
from swarm_visualizer.utility.general_utils import save_fig

from mmda.baselines.asif_core import zero_shot_classification
from mmda.utils.cca_class import NormalizedCCA
from mmda.utils.data_utils import (
    load_clip_like_data,
    load_two_encoder_data,
)
from mmda.utils.dataset_utils import (
    get_train_test_split_index,
    load_cosmos,
    load_imagenet,
    load_tiil,
    train_test_split,
)
from mmda.utils.sim_utils import (
    cosine_sim,
    roc_align_unalign_points,
    weighted_corr_sim,
)


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
        wrong_labels_bool = (
            []
        )  # True if the label is wrong, False if the label is correct
        for mturks_label_idx, orig_label_idx in zip(mturks_idx, orig_idx, strict=True):
            (
                wrong_labels_bool.append(True)
                if mturks_label_idx != orig_label_idx
                else wrong_labels_bool.append(False)
            )
        wrong_labels_bool = np.array(wrong_labels_bool, dtype=bool)
    elif cfg.dataset == "tiil":
        cfg_dataset = cfg.tiil
        img_paths, text_desciption, wrong_labels_bool, _ = load_tiil(cfg_dataset)
    elif cfg.dataset == "cosmos":
        cfg_dataset = cfg.cosmos
        img_paths, text_desciption, wrong_labels_bool, _ = load_cosmos(cfg_dataset)
    # TODO: add more datasets
    else:
        raise NotImplementedError
    return wrong_labels_bool


def separate_data(
    cfg: DictConfig,
    data1: np.ndarray,
    data2: np.ndarray,
    return_pt: bool = False,
) -> None:
    """Separate aligned data and unaligned data.

    Args:
        cfg: configuration file
        data1: data from the first encoder
        data2: data from the second encoder
        return_pt: whether to return the data as torch tensors
    Returns:
        alldata: dictionary containing the separated data
    """
    wrong_labels_bool = parse_wrong_label(cfg)
    train_idx, val_idx = get_train_test_split_index(
        cfg.train_test_ratio, data1.shape[0]
    )
    traindata1, valdata1 = train_test_split(data1, train_idx, val_idx)
    traindata2, valdata2 = train_test_split(data2, train_idx, val_idx)
    train_wrong_labels_bool, val_wrong_labels_bool = train_test_split(
        wrong_labels_bool, train_idx, val_idx
    )
    traindata1align = traindata1[~train_wrong_labels_bool]
    traindata2align = traindata2[~train_wrong_labels_bool]
    valdata1align = valdata1[~val_wrong_labels_bool]
    valdata2align = valdata2[~val_wrong_labels_bool]
    valdata1unalign = valdata1[val_wrong_labels_bool]
    valdata2unalign = valdata2[val_wrong_labels_bool]
    if return_pt:
        traindata1 = torch.tensor(traindata1).cuda()
        traindata2 = torch.tensor(traindata2).cuda()
        valdata1 = torch.tensor(valdata1).cuda()
        valdata2 = torch.tensor(valdata2).cuda()
        traindata1align = torch.tensor(traindata1align).cuda()
        traindata2align = torch.tensor(traindata2align).cuda()
        valdata1align = torch.tensor(valdata1align).cuda()
        valdata2align = torch.tensor(valdata2align).cuda()
        valdata1unalign = torch.tensor(valdata1unalign).cuda()
        valdata2unalign = torch.tensor(valdata2unalign).cuda()

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):  # noqa: ANN204, ANN002, ANN003
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    alldata = AttrDict()
    alldata.update(
        {
            "traindata1": traindata1,
            "traindata2": traindata2,
            "valdata1": valdata1,
            "valdata2": valdata2,
            "traindata1align": traindata1align,
            "traindata2align": traindata2align,
            "valdata1align": valdata1align,
            "valdata2align": valdata2align,
            "valdata1unalign": valdata1unalign,
            "valdata2unalign": valdata2unalign,
        }
    )
    return alldata


def cca_detect_mislabeled_data(cfg: DictConfig) -> list[tuple[float, float]]:
    """Detect mislabeled data using the proposed CCA method.

    Args:
        cfg: configuration file

    Returns:
        ROC points

    """
    np.random.seed(cfg.seed)
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    print(f"Loaded data1 shape: {data1.shape}, data2 shape: {data2.shape}")
    plots_path = Path(
        cfg_dataset.paths.plots_path,
        f"mislabeled_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/",
    )
    plots_path.mkdir(parents=True, exist_ok=True)

    alldata = separate_data(cfg, data1, data2)

    # select training data based on the noisy_train_set
    traindata1 = alldata.traindata1 if cfg.noisy_train_set else alldata.traindata1align
    traindata2 = alldata.traindata2 if cfg.noisy_train_set else alldata.traindata2align
    train_label = "" if cfg.noisy_train_set else "_clean"
    eq_label = "_noweight" if cfg_dataset.equal_weights else ""

    # correctly labeled case:
    cca = NormalizedCCA()
    traindata1, traindata2, corr_align = cca.fit_transform_train_data(
        cfg_dataset, traindata1, traindata2
    )

    # calculate the similarity score
    valdata1align, valdata2align = cca.transform_data(
        alldata.valdata1align, alldata.valdata2align
    )
    sim_align = weighted_corr_sim(
        valdata1align, valdata2align, corr_align, dim=cfg_dataset.sim_dim
    )

    ### mislabeled case:
    valdata1unalign, valdata2unalign = cca.transform_data(
        alldata.valdata1unalign, alldata.valdata2unalign
    )
    sim_unalign = weighted_corr_sim(
        valdata1unalign, valdata2unalign, corr_align, dim=cfg_dataset.sim_dim
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(
        data_list=[sim_align, sim_unalign],
        legend=["Aligned", "Unaligned"],
        title_str="Similarity Score Distribution",
        xlabel="Similarity Score",
        ylabel="Frequency",
        ax=ax,
        binwidth=0.05,
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
    plots_path = Path(
        cfg_dataset.paths.plots_path,
        f"mislabeled_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/",
    )
    plots_path.mkdir(parents=True, exist_ok=True)

    alldata = separate_data(cfg, data1, data2)

    sim_align = cosine_sim(alldata.valdata1align, alldata.valdata2align)
    sim_unalign = cosine_sim(alldata.valdata1unalign, alldata.valdata2unalign)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(
        data_list=[sim_align, sim_unalign],
        legend=["Aligned", "Random shuffle"],
        title_str="Similarity Score Distribution",
        xlabel="Similarity Score",
        ylabel="Frequency",
        ax=ax,
        binwidth=0.05,
    )
    save_fig(
        fig,
        plots_path
        / f"cos_similarity_mislabeled_{clip_model_name}_r{cfg.train_test_ratio}.png",
    )

    # plot ROC
    return roc_align_unalign_points(sim_align, sim_unalign, (-1, 1, 80))


def asif_detect_mislabeled_data(cfg: DictConfig) -> list[tuple[float, float]]:
    """Detect mislabeled data using the ASIF method (Paper: https://openreview.net/pdf?id=YAxV_Krcdjm).

    Args:
        cfg: configuration file

    Returns:
        ROC points
    """
    # load embeddings from the two encoders
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)

    alldata = separate_data(cfg, data1, data2, return_pt=True)

    # normalization to perform cosine similarity with a simple matmul
    alldata.traindata1 /= torch.norm(alldata.traindata1, p=2, dim=1, keepdim=True)
    alldata.traindata2 /= torch.norm(alldata.traindata2, p=2, dim=1, keepdim=True)
    alldata.valdata1 /= torch.norm(alldata.valdata1, p=2, dim=1, keepdim=True)
    alldata.valdata2 /= torch.norm(alldata.valdata2, p=2, dim=1, keepdim=True)

    # set parameters
    non_zeros = min(cfg.asif.non_zeros, alldata.traindata1.shape[0])
    range_anch = [
        2**i
        for i in range(
            int(np.log2(non_zeros) + 1), int(np.log2(len(alldata.traindata1))) + 2
        )
    ]
    range_anch = range_anch[-1:]  # run just last anchor to be quick

    # similarity score of val data
    n_anchors, scores, sims = zero_shot_classification(
        alldata.valdata1align,
        alldata.valdata2align,
        alldata.traindata1 if cfg.noisy_train_set else alldata.traindata1align,
        alldata.traindata2 if cfg.noisy_train_set else alldata.traindata2align,
        torch.zeros(alldata.valdata1align.shape[0]),
        non_zeros,
        range_anch,
        cfg.asif.val_exps,
        max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
    )
    sim_align = np.diag(sims.detach().cpu().numpy())

    # similarity score of unaligned val data
    n_anchors, scores, sims = zero_shot_classification(
        alldata.valdata1unalign,
        alldata.valdata2unalign,
        alldata.traindata1 if cfg.noisy_train_set else alldata.traindata1align,
        alldata.traindata2 if cfg.noisy_train_set else alldata.traindata2align,
        torch.zeros(alldata.valdata1unalign.shape[0]),
        non_zeros,
        range_anch,
        cfg.asif.val_exps,
        max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
    )

    sim_unalign = np.diag(sims.detach().cpu().numpy())

    # plot ROC
    return roc_align_unalign_points(sim_align, sim_unalign, (-1, 1, 150))
