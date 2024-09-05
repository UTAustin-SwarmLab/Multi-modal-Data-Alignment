"""Dataset class for classification task."""

from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig

from mmda.baselines.asif_core import zero_shot_classification
from mmda.utils.data_utils import load_clip_like_data, load_two_encoder_data
from mmda.utils.dataset_utils import (
    get_train_test_split_index,
    load_imagenet,
    load_leafy_spurge,
    shuffle_by_level,
    train_test_split,
)


class BaseClassificationDataset:
    """Base class of dataset for classification."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        self.cfg = cfg


class ImageNetDataset(BaseClassificationDataset):
    """ImageNet dataset class."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        # set random seed
        np.random.seed(cfg.seed)
        super().__init__(cfg)
        self.cfg = cfg
        _, _, self.orig_idx, self.clsidx_to_labels = load_imagenet(cfg.imagenet)

    def load_data(
        self,
        train_test_ratio: float,
        clip_bool: bool = False,
        shuffle_bool: bool = False,
    ) -> None:
        """Load the data for ImageNet dataset.

        Args:
            train_test_ratio: ratio of training data
            clip_bool: whether to use CLIP-like method
        """
        self.train_test_ratio = train_test_ratio
        if clip_bool:
            _, self.img_emb, self.text_emb = load_clip_like_data(self.cfg)
        else:
            _, self.img_emb, self.text_emb = load_two_encoder_data(self.cfg)
        train_idx, val_idx = get_train_test_split_index(
            self.train_test_ratio, self.img_emb.shape[0]
        )
        self.train_img, self.test_img = train_test_split(
            self.img_emb, train_idx, val_idx
        )
        self.train_text, self.test_text = train_test_split(
            self.text_emb, train_idx, val_idx
        )
        self.train_idx, self.test_idx = train_test_split(
            self.orig_idx, train_idx, val_idx
        )
        if shuffle_bool:
            self.train_img, self.train_text = shuffle_by_level(
                self.cfg,
                "dataset",
                self.train_img,
                self.train_text,
                self.train_idx,
                self.test_idx,
            )

    def get_labels_emb(self) -> None:
        """Get the text embeddings for all possible labels."""
        label_emb = []
        for label_idx in self.clsidx_to_labels:
            # find where the label is in the train_idx
            label_idx_in_ds = np.where(self.orig_idx == label_idx)[0]
            label_emb.append(self.text_emb[label_idx_in_ds[0]])
        self.labels_emb = np.array(label_emb)
        assert self.labels_emb.shape[0] == len(self.clsidx_to_labels)

    def classification(self, sim_fn: Union[callable, str]) -> float:  # noqa: UP007
        """Classification task.

        Args:
            sim_fn: similarity function
        Returns:
            accuracy: classification accuracy
        """
        cfg = self.cfg
        sim_scores = []
        if sim_fn == "asif":
            # set parameters
            non_zeros = min(cfg.asif.non_zeros, self.train_img.shape[0])
            range_anch = [
                2**i
                for i in range(
                    int(np.log2(non_zeros) + 1),
                    int(np.log2(len(self.train_img))) + 2,
                )
            ]
            range_anch = range_anch[-1:]  # run just last anchor to be quick
            val_labels = torch.zeros((1,), dtype=torch.float32)
            for batch_idx in range(0, self.test_img.shape[0], self.labels_emb.shape[0]):
                batch_test_img = self.test_img[
                    batch_idx : batch_idx + self.labels_emb.shape[0]
                ]
                assert (
                    batch_test_img.shape[0] == self.labels_emb.shape[0]
                ), f"{batch_test_img.shape[0]}!={self.labels_emb.shape[0]}"

                _anchors, scores, sim_score_matrix = zero_shot_classification(
                    torch.tensor(batch_test_img, dtype=torch.float32),
                    torch.tensor(self.labels_emb, dtype=torch.float32),
                    torch.tensor(self.train_img, dtype=torch.float32),
                    torch.tensor(self.train_text, dtype=torch.float32),
                    val_labels,
                    non_zeros,
                    range_anch,
                    cfg.asif.val_exps,
                    max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
                )
                sim_score_matrix = sim_score_matrix.numpy().astype(np.float32)
                sim_scores.append(sim_score_matrix)  # (batch, labels)

            sim_scores = np.concatenate(sim_scores, axis=0)  # test_img_size x labels
            sim_scores = sim_scores.T  # labels x test_img_size
            print(f"asif sim_scores shape: {sim_scores.shape}")
        else:
            for label_idx in range(self.labels_emb.shape[0]):
                print(f"Processing label {label_idx}") if label_idx % 100 == 0 else None
                label_emb = self.labels_emb[label_idx].reshape(1, -1)  # (768,)->(1,768)
                label_emb = np.repeat(label_emb, self.test_text.shape[0], axis=0)
                sim_score_matrix = sim_fn(self.test_img, label_emb)
                sim_scores.append(sim_score_matrix)
            sim_scores = np.array(sim_scores)  # labels x test_img_size

        most_similar_label_idx = np.argmax(sim_scores, axis=0)
        correct = most_similar_label_idx == self.test_idx
        return np.mean(correct)


class LeafySpurgeDataset(BaseClassificationDataset):
    """ImageNet dataset class."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.images, self.labels, self.clsidx_to_labels = load_leafy_spurge(
            cfg.leafy_spurge
        )

    def load_data(self, train_test_ratio: float, clip_bool: bool = False) -> None:
        """Load the data for ImageNet dataset.

        Args:
            train_test_ratio: ratio of training data
            clip_bool: whether to use CLIP-like method
        """
        self.train_test_ratio = train_test_ratio
        if clip_bool:
            _, self.img_emb, self.text_emb = load_clip_like_data(self.cfg)
        else:
            _, self.img_emb, self.text_emb = load_two_encoder_data(self.cfg)
        train_size = int(self.train_test_ratio * self.img_emb.shape[0])
        test_size = 100
        self.train_img, self.test_img = (
            self.img_emb[:train_size],
            self.img_emb[-test_size:],
        )
        self.train_text, self.test_text = (
            self.text_emb[:train_size],
            self.text_emb[-test_size:],
        )
        self.train_idx, self.test_idx = (
            self.labels[:train_size],
            self.labels[-test_size:],
        )

    def get_labels_emb(self) -> None:
        """Get the text embeddings for all possible labels."""
        label_emb = []
        self.labels_np = np.array(self.labels)
        for label_idx in self.clsidx_to_labels:
            # find where the label is in the train_idx
            label_idx_in_ds = np.where(self.labels_np == label_idx)[0]
            label_emb.append(self.text_emb[label_idx_in_ds[0]])
        self.labels_emb = np.array(label_emb)
        assert self.labels_emb.shape[0] == len(self.clsidx_to_labels)

    def classification(self, sim_fn: Union[callable, str]) -> float:  # noqa: UP007
        """Classification task.

        Args:
            sim_fn: similarity function
        Returns:
            accuracy: classification accuracy
        """
        cfg = self.cfg
        sim_scores = []
        if sim_fn == "asif":
            # set parameters
            non_zeros = min(cfg.asif.non_zeros, self.train_img.shape[0])
            range_anch = [
                2**i
                for i in range(
                    int(np.log2(non_zeros) + 1),
                    int(np.log2(len(self.train_img))) + 2,
                )
            ]
            range_anch = range_anch[-1:]  # run just last anchor to be quick
            val_labels = torch.zeros((1,), dtype=torch.float32)
            # generate noise in the shape of the labels_emb
            noise = np.random.rand(
                self.test_img.shape[0] - self.labels_emb.shape[0],
                self.labels_emb.shape[1],
            ).astype(np.float32)
            self.test_label = np.concatenate((self.labels_emb, noise), axis=0)
            assert (
                self.test_img.shape[0] == self.test_label.shape[0]
            ), f"{self.test_img.shape[0]}!={self.test_label.shape[0]}"
            _anchors, scores, sim_score_matrix = zero_shot_classification(
                torch.tensor(self.test_img, dtype=torch.float32),
                torch.tensor(self.test_label, dtype=torch.float32),
                torch.tensor(self.train_img, dtype=torch.float32),
                torch.tensor(self.train_text, dtype=torch.float32),
                val_labels,
                non_zeros,
                range_anch,
                cfg.asif.val_exps,
                max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
            )
            sim_score_matrix = sim_score_matrix.numpy().astype(np.float32)[:, :2]
            sim_scores = sim_score_matrix.T  # labels x test_img_size
        else:
            for label_idx in range(self.labels_emb.shape[0]):
                print(f"Processing label {label_idx}") if label_idx % 100 == 0 else None
                label_emb = self.labels_emb[label_idx].reshape(1, -1)  # (768,)->(1,768)
                label_emb = np.repeat(label_emb, self.test_text.shape[0], axis=0)
                sim_score_matrix = sim_fn(self.test_img, label_emb)
                sim_scores.append(sim_score_matrix)
            sim_scores = np.array(sim_scores)  # labels x test_img_size

        most_similar_label_idx = np.argmax(sim_scores, axis=0)
        correct = most_similar_label_idx == self.test_idx
        return np.mean(correct)


def load_classification_dataset(cfg: DictConfig) -> ImageNetDataset:
    """Load the dataset for classification task.

    Args:
        cfg: configuration file
    Returns:
        dataset: dataset class
    """
    if cfg.dataset == "imagenet":
        dataset = ImageNetDataset(cfg)
    elif cfg.dataset == "leafy_spurge":
        dataset = LeafySpurgeDataset(cfg)
    else:
        msg = f"Dataset {cfg.dataset} not supported"
        raise ValueError(msg)
    return dataset
