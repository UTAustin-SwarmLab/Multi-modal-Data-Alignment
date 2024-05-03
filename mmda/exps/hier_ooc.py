"""Use a hierarchical decision making process to predict out-of-context image and captions.

The setting is, given a set of corresponding images and captions, predict the out-of-context of a new caption.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from swarm_visualizer.utility.general_utils import save_fig

from mmda.baselines.asif_core import zero_shot_classification
from mmda.utils.cca_class import NormalizedCCA
from mmda.utils.data_utils import load_clip_like_data, load_two_encoder_data
from mmda.utils.ooc_dataset_class import load_hier_dataset
from mmda.utils.sim_utils import cosine_sim, weighted_corr_sim


def cca_hier_ooc(cfg: DictConfig) -> list[(float, float)]:
    """Hierarchical decision making process to predict out-of-context image and captions using the proposed CCA method.

    Args:
        cfg: configuration file
    Returns:
        dict of (txt_threshold, text_img_threshold): (tp, fp, fn, tn)
    """
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    hier_ds = load_hier_dataset(cfg)
    hier_ds.split_data(data1, data2)
    # CCA transformation of img and text
    eq_label = "_noweight" if cfg[cfg.dataset].equal_weights else ""
    cca_save_path = Path(cfg_dataset.paths.save_path) / (
        f"ooc_cca_model_size{hier_ds.train_gt_img_emb.shape[0]}_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}{eq_label}.pkl"
    )
    cca = NormalizedCCA()
    if not cca_save_path.exists():
        cfg_dataset.sim_dim = min(
            hier_ds.train_gt_img_emb.shape[1], hier_ds.train_gt_text_emb.shape[1]
        )
        print(f"Fit the CCA model. CCA dimension: {cfg_dataset.sim_dim}")
        print(
            f"Train data shape: {hier_ds.train_gt_img_emb.shape}, {hier_ds.train_gt_text_emb.shape}"
        )
        hier_ds.train_gt_img_emb, hier_ds.train_gt_text_emb, corr = (
            cca.fit_transform_train_data(
                cfg_dataset, hier_ds.train_gt_img_emb, hier_ds.train_gt_text_emb
            )
        )
        cca.save_model(cca_save_path)  # save the class object for later use
        print(f"Save the CCA model to {cca_save_path}.")
    else:
        print(f"Load the CCA model from {cca_save_path}.")
        cca.load_model(cca_save_path)
        hier_ds.train_gt_img_emb = cca.traindata1
        hier_ds.train_gt_text_emb = cca.traindata2
        corr = cca.corr_coeff

    if not cfg_dataset.equal_weights:
        plots_path = (
            Path(cfg_dataset.paths.plots_path)
            / f"hier_ooc_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}_{cfg_dataset.detection_rule}/"
        )
        plots_path.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(corr)
        ax.set_title("Correlation Coefficients of the Cross Covariance")
        ax.set_xlabel("Dimension of Eigenvalues")
        ax.set_ylabel("Correlation Coefficients")
        ax.set_ylim(0, 1)
        save_fig(fig, plots_path / "correlation_coefficient.png")

    hier_ds.test_gt_img_emb, hier_ds.test_gt_text_emb = cca.transform_data(
        hier_ds.test_gt_img_emb, hier_ds.test_gt_text_emb
    )
    hier_ds.test_new_img_emb, hier_ds.test_new_text_emb = cca.transform_data(
        hier_ds.test_new_img_emb, hier_ds.test_new_text_emb
    )

    def text_img_sim_fn(x: np.array, y: np.array) -> np.array:
        return weighted_corr_sim(x, y, corr=corr, dim=cfg_dataset.sim_dim)

    hier_ds.set_similarity_metrics(cosine_sim, text_img_sim_fn)
    return hier_ds.detect_ooc()  # tp, fp, fn, tn


def clip_like_hier_ooc(cfg: DictConfig) -> list[(float, float)]:
    """Hierarchical decision making process to predict out-of-context image and captions using the CLIP-like method.

    Args:
        cfg: configuration file
    Returns:
        dict of (txt_threshold, text_img_threshold): (tp, fp, fn, tn)
    """
    cfg_dataset, data1, data2 = load_clip_like_data(cfg)
    hier_ds = load_hier_dataset(cfg)
    hier_ds.split_data(data1, data2)
    hier_ds.set_similarity_metrics(cosine_sim, cosine_sim)
    return hier_ds.detect_ooc()  # tp, fp, fn, tn


def asif_hier_ooc(cfg: DictConfig) -> list[(float, float)]:
    """Hierarchical decision making process to predict out-of-context image and captions using the ASIF method.

    Args:
        cfg: configuration file
    Returns:
        dict of (txt_threshold, text_img_threshold): (tp, fp, fn, tn)
    """
    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    hier_ds = load_hier_dataset(cfg)
    hier_ds.split_data(data1, data2)

    # normalization to perform cosine similarity with a simple matmul
    hier_ds.train_gt_img_emb /= np.linalg.norm(hier_ds.train_gt_img_emb, axis=0)
    hier_ds.train_gt_text_emb /= np.linalg.norm(hier_ds.train_gt_text_emb, axis=0)
    hier_ds.test_gt_img_emb /= np.linalg.norm(hier_ds.test_gt_img_emb, axis=0)
    hier_ds.test_gt_text_emb /= np.linalg.norm(hier_ds.test_gt_text_emb, axis=0)
    hier_ds.test_new_img_emb /= np.linalg.norm(hier_ds.test_new_img_emb, axis=0)
    hier_ds.test_new_text_emb /= np.linalg.norm(hier_ds.test_new_text_emb, axis=0)

    # set parameters
    non_zeros = min(cfg.asif.non_zeros, hier_ds.train_gt_img_emb.shape[0])
    range_anch = [
        2**i
        for i in range(
            int(np.log2(non_zeros) + 1), int(np.log2(len(hier_ds.train_gt_img_emb))) + 2
        )
    ]
    range_anch = range_anch[-1:]  # run just last anchor to be quick

    def text_img_sim_fn(x: np.array, y: np.array) -> np.array:
        n_anchors, scores, sims = zero_shot_classification(
            torch.tensor(x).cuda(),
            torch.tensor(y).cuda(),
            torch.tensor(hier_ds.train_gt_img_emb).cuda(),
            torch.tensor(hier_ds.train_gt_text_emb).cuda(),
            torch.zeros(x.shape[0]),
            non_zeros,
            range_anch,
            cfg.asif.val_exps,
            max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
        )
        return np.diag(sims.detach().cpu().numpy())

    hier_ds.set_similarity_metrics(cosine_sim, text_img_sim_fn)
    return hier_ds.detect_ooc()  # tp, fp, fn, tn
