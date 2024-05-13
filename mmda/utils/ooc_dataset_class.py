"""Dataset class for retrieval task."""

import numpy as np
from omegaconf import DictConfig

from mmda.utils.dataset_utils import load_cosmos
from mmda.utils.roc_utils import (
    convex_hull_roc_points,
    select_maximum_auc,
    tp_fp_fn_tn_to_roc,
)


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
        self.linspace = np.linspace(-1, 1, 80)
        self.detection_rule = cfg[cfg.dataset].detection_rule
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

    def detect_ooc(self) -> list[(float, float)]:
        """Detect out-of-context data.

        Returns:
            roc_points: ROC points
        """
        self.get_texts_similarity()
        self.get_text_image_similarity()
        if self.detection_rule == "bilevel":
            detection_results = self.bilevel_detect_ooc()
            if self.text_text_sim_fn == self.text_img_sim_fn:
                roc_points = select_maximum_auc(detection_results)
            else:
                roc_points = convex_hull_roc_points(detection_results)
        elif self.detection_rule == "mean":
            tps = self.mean_detect_ooc()
            roc_points = tp_fp_fn_tn_to_roc(tps)
        else:
            msg = f"Detection rule {self.detection_rule} not supported"
            raise ValueError(msg)
        return roc_points

    def mean_detect_ooc(
        self,
    ) -> list[(float, float)]:
        """Detect out-of-context data.

        We have the similarity scores for text-text and text-image.
        We use the mean of the two similarities as the detection threshold.
        mean = (text-text + text-image) / 2
        if mean < threshold, it is out of context.
        else, it is in context.

        Returns:
            detection_results: (texts_threshold, text_image_threshold): (tp, fp, fn, tn)
        """
        mean_sim = (self.texts_sim + self.text_image_sim) / 2
        roc_points = []
        for threshold in self.linspace:  # compare C1 and C2
            mean_ooc_mask = self.filter_ooc_data(threshold, mean_sim)
            tp = np.sum(mean_ooc_mask & self.test_new_wrong_mask)
            fp = np.sum(mean_ooc_mask & ~self.test_new_wrong_mask)
            fn = np.sum(~mean_ooc_mask & self.test_new_wrong_mask)
            tn = np.sum(~mean_ooc_mask & ~self.test_new_wrong_mask)
            roc_points.append((tp, fp, fn, tn))
        return roc_points

    def bilevel_detect_ooc(
        self,
    ) -> dict[tuple[float, float], tuple[float, float, float, float]]:
        """Detect out-of-context data.

        We have the similarity scores for text-text and text-image. We run a two-level detection of OOC data.
        The first level is to detect OOC text data. The second level is to detect OOC image data.
        Ground truth: C1=Image aligned
        if C1=C2
          C2!=I -> in context
          C2=I -> in context
        else C1!=C2
          C2!=I -> out of context
          C2=I -> in context

        Returns:
            detection_results: (texts_threshold, text_image_threshold): (tp, fp, fn, tn)
        """
        ooc_texts_mask_dict = {}
        ooc_text_image_mask_dict = {}
        detection_results = {}
        for texts_threshold in self.linspace:  # compare C1 and C2
            ooc_texts_mask = self.filter_ooc_data(texts_threshold, self.texts_sim)
            ooc_texts_mask_dict[texts_threshold] = ooc_texts_mask
        for text_image_threshold in self.linspace:  # compare C2 and I
            ooc_text_image_mask = self.filter_ooc_data(
                text_image_threshold, self.text_image_sim
            )
            ooc_text_image_mask_dict[text_image_threshold] = ooc_text_image_mask
        for texts_threshold in self.linspace:
            for text_image_threshold in self.linspace:
                text_img_not_align = (
                    ooc_texts_mask_dict[texts_threshold]
                    & ooc_text_image_mask_dict[text_image_threshold]
                )  # C1!=C2 & C2!=I
                image_not_align = ooc_text_image_mask_dict[
                    text_image_threshold
                ]  # C2!=I
                two_level_ooc_mask = image_not_align & text_img_not_align
                tp = np.sum(two_level_ooc_mask & self.test_new_wrong_mask)
                fp = np.sum(two_level_ooc_mask & ~self.test_new_wrong_mask)
                fn = np.sum(~two_level_ooc_mask & self.test_new_wrong_mask)
                tn = np.sum(~two_level_ooc_mask & ~self.test_new_wrong_mask)
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
