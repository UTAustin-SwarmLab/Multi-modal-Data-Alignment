"""Dataset class for retrieval task."""

import numpy as np
from omegaconf import DictConfig

from mmda.utils.dataset_utils import load_cosmos


# define base class of dataset
class BaseOocDataset:
    """Base class of dataset for out-of-context detection."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        self.cfg = cfg
        self.sim_dim = cfg[cfg.dataset].sim_dim
        self.text_img_sim_fn = None
        self.text_text_sim_fn = None

    def split_data(self, data1: np.ndarray, data2: np.ndarray) -> None:
        """Split the data. Check the shape of the data.

        Args:
            data1: data from the first encoder
            data2: data from the second encoder
        """
        assert data1.shape[0] == data2.shape[0], f"{data1.shape[0]}!={data2.shape[0]}"


class COSMOSOocDataset(BaseOocDataset):
    """COSMOS dataset class for out-of-context detection."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        super().__init__(cfg)
        self.img_path, self.txt_descriptions, self.wrong_label, _ = load_cosmos(
            cfg[cfg.dataset]
        )
        self.train_gt_idx = np.array(range(len(self.img_path)))[:-3400]
        # idx for in context test ground truth data
        self.test_gt_idx = np.array(range(len(self.img_path)))[-3400:][::2]
        # idx for possibly out of context test data
        self.test_new_idx = np.array(range(len(self.img_path)))[-3400:][1::2]
        assert len(self.train_gt_idx) + len(self.test_gt_idx) + len(
            self.test_new_idx
        ) == len(self.img_path), "Sum of data split error"
        assert len(self.test_gt_idx) == len(
            self.test_new_idx
        ), f"Test ata split error, {len(self.test_gt_idx)}!={len(self.test_new_idx)}"

        self.train_gt_wrong_mask = self.wrong_label[self.train_gt_idx]
        self.test_new_wrong_mask = self.wrong_label[self.test_new_idx]
        assert np.sum(self.train_gt_wrong_mask) == 0, "Train gt data has wrong labels"
        assert (
            np.sum(self.test_new_wrong_mask) == self.test_new_wrong_mask.shape[0] / 2
        ), "Out-of-context labels should be half of the test data"

    def split_data(self, data1: np.ndarray, data2: np.ndarray) -> None:
        """Split the data. Split the data into training and testing sets."""
        super().split_data(data1, data2)
        self.train_gt_img_emb = data1[self.train_gt_idx]
        self.train_gt_text_emb = data2[self.train_gt_idx]
        self.test_gt_img_emb = data1[self.test_gt_idx]
        self.test_gt_text_emb = data2[self.test_gt_idx]
        self.test_new_img_emb = data1[self.test_new_idx]
        self.test_new_text_emb = data2[self.test_new_idx]
        assert (
            self.test_new_text_emb.shape[0] == self.test_new_wrong_mask.shape[0]
        ), f"Test data and mask shape mismatch, {self.test_new_text_emb.shape[0]}!={self.test_new_wrong_mask.shape[0]}"

    def set_similarity_metrics(
        self, text_text_sim_fn: callable, text_img_sim_fn: callable
    ) -> None:
        """Set the similarity function.

        Args:
            text_text_sim_fn: similarity function between texts
            text_img_sim_fn: similarity function between text and image
        """
        self.text_img_sim_fn = text_img_sim_fn
        self.text_text_sim_fn = text_text_sim_fn

    def bilevel_detect_ooc(self) -> None:
        """Detect out-of-context data.

        We have the similarity scores for text-text and text-image so we run a two-level detection of OOC data.
        The first level is to detect OOC text data. The second level is to detect OOC image data.
        Ground truth: C1=Image aligned
        if C1=C2
          C2!=I -> in or out of context?
          C2=I -> in context
        else C1!=C2
          C2!=I -> out of context
          C2=I -> in context
        """
        self.get_texts_similarity()
        self.get_text_image_similarity()
        ooc_ground_truth = self.test_new_wrong_mask
        ooc_texts_mask_dict = {}
        ooc_text_image_mask_dict = {}
        detection_results = {}
        for texts_threshold in np.linspace(-1, 1, 40):  # compare C1 and C2
            ooc_texts_mask = self.filter_ooc_data(texts_threshold, self.texts_sim)
            ooc_texts_mask_dict[texts_threshold] = ooc_texts_mask
        for text_image_threshold in np.linspace(-1, 1, 40):  # compare C2 and I
            ooc_text_image_mask = self.filter_ooc_data(
                text_image_threshold, self.text_image_sim
            )
            ooc_text_image_mask_dict[text_image_threshold] = ooc_text_image_mask
        for texts_threshold in np.linspace(-1, 1, 40):
            for text_image_threshold in np.linspace(-1, 1, 40):
                text_img_not_align = (
                    ooc_texts_mask_dict[texts_threshold]
                    & ooc_text_image_mask_dict[text_image_threshold]
                )  # C1!=C2 & C2!=I
                image_not_align = ooc_text_image_mask_dict[
                    text_image_threshold
                ]  # C2!=I
                two_level_ooc_mask = image_not_align & text_img_not_align
                tp = np.sum(two_level_ooc_mask & ooc_ground_truth)
                fp = np.sum(two_level_ooc_mask & ~ooc_ground_truth)
                fn = np.sum(~two_level_ooc_mask & ooc_ground_truth)
                tn = np.sum(~two_level_ooc_mask & ~ooc_ground_truth)
                detection_results[(texts_threshold, text_image_threshold)] = (
                    tp,
                    fp,
                    fn,
                    tn,
                )
        return detection_results

    def get_texts_similarity(self) -> None:
        """Compare the text embeddings."""
        assert self.text_text_sim_fn is not None, "text_text_sim_fn is not set"
        self.texts_sim = self.text_text_sim_fn(
            self.test_gt_text_emb, self.test_new_text_emb
        )

    def get_text_image_similarity(self) -> None:
        """Compare the image embeddings."""
        assert self.text_img_sim_fn is not None, "text_img_sim_fn is not set"
        self.text_image_sim = self.text_img_sim_fn(
            self.test_gt_img_emb, self.test_new_text_emb
        )

    def filter_ooc_data(
        self, threshold: float, sim_scores: np.ndarray, less_than: bool = True
    ) -> np.ndarray:
        """Filter out-of-context data.

        Args:
            threshold: threshold for filtering
            sim_scores: similarity scores
            less_than: True if the threshold is less than (not similar), False otherwise.

        Returns:
            bool_idx: boolean index for out-of-context data
        """
        return (sim_scores <= threshold) if less_than else (sim_scores > threshold)


def load_hier_dataset(cfg: DictConfig) -> COSMOSOocDataset:
    """Load the dataset for retrieval task.

    Args:
        cfg: configuration file
    Returns:
        dataset: dataset class
    """
    if cfg.dataset == "cosmos":
        dataset = COSMOSOocDataset(cfg)
    else:
        msg = f"Dataset {cfg.dataset} not supported"
        raise ValueError(msg)
    return dataset
