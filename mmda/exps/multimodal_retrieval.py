"""This module contains the functions to detect mislabeled data using the proposed method and baselines."""

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from mmda.baselines.asif_core import zero_shot_classification
from mmda.utils.cca_utils import cca_fit_train_data
from mmda.utils.data_utils import (
    load_clip_like_data,
    load_two_encoder_data,
    origin_centered,
)
from mmda.utils.dataset_utils import (
    load_flickr,
)
from mmda.utils.sim_utils import (
    cosine_sim,
    weighted_corr_sim,
)


def preprocess_retrieval_data(
    cfg: DictConfig, data1: np.ndarray, data2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Preprocess the data for retrieval.

    Args:
        cfg: configuration file
        data1: data from the first encoder
        data2: data from the second encoder
    Returns:
        traindata1: training data from the first encoder
        traindata2: training data from the second encoder
        testdata1: test data from the first encoder
        testdata2: test data from the second encoder
        test_img_ids: test image ids
        total_num_gt: total number of ground truth
    """
    if cfg.dataset == "flickr":
        img_path, txt_descriptions, splits, img_ids = load_flickr(cfg["flickr"])
        if not cfg["flickr"].img2text:  # text to retrieve image
            # data1 is the input modal and data2 is the output modal
            # data 1 is originally image and data2 is originally text
            data1, data2 = data2, data1
            total_num_gt = 1  # Total number of relevant images in the database
        else:
            total_num_gt = 5  # Total number of relevant texts in the database
    # TODO: add more datasets
    else:
        raise NotImplementedError

    num_data = data1.shape[0]
    assert len(img_path) == num_data, f"{len(img_path)} != {num_data}"
    assert len(txt_descriptions) == num_data, f"{len(txt_descriptions)} != {num_data}"
    assert len(splits) == num_data, f"{len(splits)} != {num_data}"
    assert len(img_ids) == num_data, f"{len(img_ids)} != {num_data}"

    train_idx = np.where(splits == "train")[0]
    test_idx = np.where(splits == "test")[0]

    traindata1, traindata2 = data1[train_idx], data2[train_idx]
    testdata1, testdata2 = data1[test_idx], data2[test_idx]
    test_img_ids = img_ids[test_idx]

    # zero mean data
    traindata1, traindata1_mean = origin_centered(traindata1)
    traindata2, traindata2_mean = origin_centered(traindata2)
    testdata1 = testdata1 - traindata1_mean
    testdata2 = testdata2 - traindata2_mean
    # make sure the data is zero mean
    assert np.allclose(
        traindata1.mean(axis=0), 0, atol=1e-4
    ), f"traindata1 not zero mean: {traindata1.mean(axis=0)}"
    assert np.allclose(
        traindata2.mean(axis=0), 0, atol=1e-4
    ), f"traindata2 not zero mean: {traindata2.mean(axis=0)}"
    return traindata1, traindata2, testdata1, testdata2, test_img_ids, total_num_gt


def cca_retrieval(cfg: DictConfig) -> tuple[dict[float:float], dict[float:float]]:
    """Retrieve data using the proposed CCA method.

    Args:
        cfg: configuration file
    Returns:
        recalls: {1: recall@1, 5:recall@5} if img2text else {1:recall@1}
        precisions: {1: precision@1, 5:precision@5} if img2text else {1:precision@1}
    """
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    print(f"Loaded data1 shape: {data1.shape}, data2 shape: {data2.shape}")
    plots_path = Path(
        cfg_dataset.paths.plots_path,
        f"retrieval_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/",
    )
    plots_path.mkdir(parents=True, exist_ok=True)

    traindata1, traindata2, testdata1, testdata2, test_img_ids, total_num_gt = (
        preprocess_retrieval_data(cfg, data1, data2)
    )

    cca, traindata1, traindata2, corr = cca_fit_train_data(
        cfg_dataset, traindata1, traindata2
    )

    # calculate the similarity score
    testdata1, testdata2 = cca.transform((testdata1, testdata2))
    recalls, precisions = [], []
    # retrieval
    for idx in range(testdata1.shape[0]):
        gt_img_id = test_img_ids[idx]
        test_datapoint = testdata1[idx, :].reshape(1, -1)
        # copy the test text to the number of images
        test_text_emb = np.repeat(test_datapoint, testdata2.shape[0], axis=0)
        sim = weighted_corr_sim(test_text_emb, testdata2, corr, dim=cfg_dataset.sim_dim)
        # sort the similarity score in descending order and get the index
        sim_idx = np.argsort(sim)[::-1]
        # calculate the recall
        # Recall = Total number of correct data retrieved/Total number of relevant documents in the database
        hit = np.zeros(total_num_gt)
        for ii in range(total_num_gt):
            hit[ii] = 1 if gt_img_id == test_img_ids[sim_idx[ii]] else 0
        # calculate the precision
        recall = np.cumsum(hit) / total_num_gt
        # Precision = Total number of correct data retrieved/Total number of retrieved documents
        precision = np.cumsum(hit) / (np.arange(total_num_gt) + 1)
        recalls.append(recall)
        precisions.append(precision)

    recalls = np.array(recalls).mean(axis=0)
    precisions = np.array(precisions).mean(axis=0)

    return (
        {1: recalls[0]},
        {1: precisions[0]} if total_num_gt == 1 else {1: recalls[0], 5: recalls[4]},
        {1: precisions[0], 5: precisions[4]},
    )


def clip_like_retrieval(cfg: DictConfig) -> tuple[dict[float:float], dict[float:float]]:
    """Retrieve data using the CLIP-like method.

    Args:
        cfg: configuration file
    Returns:
        recalls: {1: recall@1, 5:recall@5} if img2text else {1:recall@1}
        precisions: {1: precision@1, 5:precision@5} if img2text else {1:precision@1}
    """
    cfg_dataset, data1, data2 = load_clip_like_data(cfg)
    clip_model_name = "CLAP" if cfg.dataset == "musiccaps" else "CLIP"
    print(f"Loaded data1 shape: {data1.shape}, data2 shape: {data2.shape}")
    plots_path = Path(
        cfg_dataset.paths.plots_path,
        f"retrieval_{clip_model_name}_{clip_model_name}/",
    )
    plots_path.mkdir(parents=True, exist_ok=True)

    traindata1, traindata2, testdata1, testdata2, test_img_ids, total_num_gt = (
        preprocess_retrieval_data(cfg, data1, data2)
    )

    # calculate the similarity score
    recalls, precisions = [], []
    for idx in range(testdata1.shape[0]):
        gt_img_id = test_img_ids[idx]
        test_datapoint = testdata1[idx, :].reshape(1, -1)
        # copy the test text to the number of images
        test_text_emb = np.repeat(test_datapoint, testdata2.shape[0], axis=0)
        sim = cosine_sim(test_text_emb, testdata2)
        # sort the similarity score in descending order and get the index
        sim_idx = np.argsort(sim)[::-1]
        # calculate the recall
        # Recall = Total number of correct data retrieved/Total number of relevant documents in the database
        hit = np.zeros(total_num_gt)
        for ii in range(total_num_gt):
            hit[ii] = 1 if gt_img_id == test_img_ids[sim_idx[ii]] else 0
        # calculate the precision
        recall = np.cumsum(hit) / total_num_gt
        # Precision = Total number of correct data retrieved/Total number of retrieved documents
        precision = np.cumsum(hit) / (np.arange(total_num_gt) + 1)
        recalls.append(recall)
        precisions.append(precision)

    recalls = np.array(recalls).mean(axis=0)
    precisions = np.array(precisions).mean(axis=0)

    return (
        {1: recalls[0]},
        {1: precisions[0]} if total_num_gt == 1 else {1: recalls[0], 5: recalls[4]},
        {1: precisions[0], 5: precisions[4]},
    )


def asif_retrieval(
    cfg: DictConfig,
) -> tuple[dict[float:float], dict[float:float]]:
    """Retrieve data using the ASIF method.

    Paper: https://openreview.net/pdf?id=YAxV_Krcdjm
    Args:
        cfg: configuration file
    Returns:
        recalls: {1: recall@1, 5:recall@5} if img2text else {1:recall@1}
        precisions: {1: precision@1, 5:precision@5} if img2text else {1:precision@1}
    """
    cfg_dataset, data1, data2 = load_clip_like_data(cfg)
    clip_model_name = "CLAP" if cfg.dataset == "musiccaps" else "CLIP"
    print(f"Loaded data1 shape: {data1.shape}, data2 shape: {data2.shape}")
    plots_path = Path(
        cfg_dataset.paths.plots_path,
        f"retrieval_{clip_model_name}_{clip_model_name}/",
    )
    plots_path.mkdir(parents=True, exist_ok=True)

    traindata1, traindata2, testdata1, testdata2, test_img_ids, total_num_gt = (
        preprocess_retrieval_data(cfg, data1, data2)
    )

    # set parameters
    non_zeros = min(cfg.asif.non_zeros, traindata1.shape[0])
    range_anch = [
        2**i
        for i in range(int(np.log2(non_zeros) + 1), int(np.log2(len(traindata1))) + 2)
    ]
    range_anch = range_anch[-1:]  # run just last anchor to be quick

    # calculate the similarity score
    recalls, precisions = [], []
    for idx in range(testdata1.shape[0]):
        gt_img_id = test_img_ids[idx]
        test_datapoint = testdata1[idx, :].reshape(1, -1)
        # copy the test text to the number of images
        test_text_emb = np.repeat(test_datapoint, testdata2.shape[0], axis=0)

        # similarity score of val data
        n_anchors, scores, sims = zero_shot_classification(
            test_text_emb,
            testdata2,
            traindata1,
            traindata2,
            torch.zeros(test_text_emb.shape[0]),
            non_zeros,
            range_anch,
            cfg.asif.val_exps,
            max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
        )
        sim = np.diag(sims.detach().cpu().numpy())
        # sort the similarity score in descending order and get the index
        sim_idx = np.argsort(sim)[::-1]
        # calculate the recall
        # Recall = Total number of correct data retrieved/Total number of relevant documents in the database
        hit = np.zeros(total_num_gt)
        for ii in range(total_num_gt):
            hit[ii] = 1 if gt_img_id == test_img_ids[sim_idx[ii]] else 0
        # calculate the precision
        recall = np.cumsum(hit) / total_num_gt
        # Precision = Total number of correct data retrieved/Total number of retrieved documents
        precision = np.cumsum(hit) / (np.arange(total_num_gt) + 1)
        recalls.append(recall)
        precisions.append(precision)

    recalls = np.array(recalls).mean(axis=0)
    precisions = np.array(precisions).mean(axis=0)

    return (
        {1: recalls[0]},
        {1: precisions[0]} if total_num_gt == 1 else {1: recalls[0], 5: recalls[4]},
        {1: precisions[0], 5: precisions[4]},
    )
