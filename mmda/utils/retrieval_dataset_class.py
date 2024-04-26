"""Dataset class for retrieval task."""

import numpy as np
from omegaconf import DictConfig

from mmda.utils.dataset_utils import load_flickr


# define base class of dataset
class BaseRetrievalDataset:
    """Base class of dataset for retrieval."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        self.cfg = cfg
        self.img_path = None
        self.txt_descriptions = None
        self.num_gt = None
        self.img2text = None
        self.test_img_ids = None

    def preprocess_retrieval_data(self, data1: np.ndarray, data2: np.ndarray) -> None:
        """Preprocess the data for retrieval. Check the shape of the data.

        Args:
            data1: data from the first encoder
            data2: data from the second encoder
        """
        assert data1.shape[0] == data2.shape[0], f"{data1.shape[0]}!={data2.shape[0]}"

    def map_precision_similarity(
        self, sim_fn: callable
    ) -> tuple[float, dict[float:float]]:
        """Calculate the mean average precision and precision at k (1 ~ num_gt).

        Args:
            sim_fn: similarity function
        Returns:
            map: {mAP}
            precisions: {1: precision@1, 5:precision@5}
            similarity: similarity score
        """
        maps, precisions = [], []
        sim_scores = []
        for idx in range(0, self.testdata1.shape[0], self.num_gt):
            gt_img_id = self.test_img_ids[idx]
            test_datapoint = self.testdata1[idx, :].reshape(1, -1)
            # copy the test text to the number of images
            test_text_emb = np.repeat(test_datapoint, self.testdata2.shape[0], axis=0)
            sim_score = sim_fn(test_text_emb, self.testdata2)
            # sort the similarity score in descending order and get the index
            sim_top_idx = np.argpartition(sim_score, -self.num_gt)[-self.num_gt :]
            sim_top_idx = sim_top_idx[np.argsort(sim_score[sim_top_idx])[::-1]]
            hit = np.zeros(5)
            for ii in range(self.num_gt):
                hit[ii] = 1 if gt_img_id == self.test_img_ids[sim_top_idx[ii]] else 0
            # Precision = Total number of correct data retrieved/Total number of retrieved documents
            precision = np.cumsum(hit) / (np.arange(5) + 1)  # array
            # average precision
            ap = 1 / self.num_gt * np.sum(precision * hit)  # scalar
            maps.append(ap)
            precisions.append(precision)
            sim_scores.append(sim_score)
        maps = np.array(maps).mean()
        precisions = np.array(precisions).mean(axis=0)
        sim_scores = np.concatenate(sim_scores, axis=0)
        return maps, {1: precisions[0], 5: precisions[4]}, sim_scores

    def top_k_presicion(self, sim_fn: callable) -> tuple[float, dict[float:float]]:
        """Calculate the average precision and precision at k (1 ~ num_gt).

        Args:
            sim_fn: similarity function
        Returns:
            map: {mAP}
            precisions: {1: precision@1, 5:precision@5}
        """
        return self.map_precision_similarity(sim_fn)[0:2]


class FlickrDataset(BaseRetrievalDataset):
    """Flickr dataset class."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.img_path, self.txt_descriptions, self.splits, self.img_ids = load_flickr(
            cfg["flickr"]
        )
        self.img2text = cfg["flickr"].img2text

    def preprocess_retrieval_data(self, data1: np.ndarray, data2: np.ndarray) -> None:
        """Preprocess the data for retrieval.

        Args:
            data1: data from the first encoder
            data2: data from the second encoder
        """
        # remove redundant text descriptions
        assert data1.shape[0] == data2.shape[0], f"{data1.shape[0]}!={data2.shape[0]}"
        assert (
            data1.shape[0] == self.splits.shape[0]
        ), f"{data1.shape[0]}!={self.splits.shape[0]}"
        assert (
            data1.shape[0] == self.img_ids.shape[0]
        ), f"{data1.shape[0]}!={self.img_ids.shape[0]}"

        super().preprocess_retrieval_data(data1, data2)
        if self.img2text:  # image to retrieve text
            self.data1, self.data2 = data1, data2
            self.num_gt = 5  # Total number of relevant texts in the database
        else:  # text to retrieve image
            # data 1 is originally image and data2 is originally text
            self.data1, self.data2 = data2, data1
            self.num_gt = 1  # Total number of relevant images in the database

        self.train_idx = np.where(self.splits == "train")[0]  # 145_000
        self.test_idx = np.where(self.splits == "test")[0]  # 5_000

        self.traindata1, self.traindata2 = (
            self.data1[self.train_idx],
            self.data2[self.train_idx],
        )
        self.testdata1, self.testdata2 = (
            self.data1[self.test_idx],
            self.data2[self.test_idx],
        )
        self.test_img_ids = self.img_ids[self.test_idx]


def load_retrieval_dataset(cfg: DictConfig) -> FlickrDataset:
    """Load the dataset for retrieval task.

    Args:
        cfg: configuration file
    Returns:
        dataset: dataset class
    """
    if cfg.dataset == "flickr":
        dataset = FlickrDataset(cfg)
    else:
        msg = f"Dataset {cfg.dataset} not supported"
        raise ValueError(msg)
    return dataset
