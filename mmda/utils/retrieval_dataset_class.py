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
        self.total_num_gt = None
        self.img2text = None
        self.test_img_ids = None

    def preprocess_retrieval_data(self, data1: np.ndarray, data2: np.ndarray) -> None:
        """Preprocess the data for retrieval. Check the shape of the data.

        Args:
            data1: data from the first encoder
            data2: data from the second encoder
        """
        assert (
            data1.shape[0] == data2.shape[0]
        ), f"Data shape mismatch. {data1.shape[0]} != {data2.shape[0]}"

    def recall_presicion_at_k(
        self, sim_fn: callable
    ) -> tuple[dict[float:float], dict[float:float]]:
        """Calculate the recall and precision at k (1 ~ total_num_gt).

        Args:
            sim_fn: similarity function
        Returns:
            recalls: {1: recall@1, 5:recall@5} if img2text else {1:recall@1}
            precisions: {1: precision@1, 5:precision@5} if img2text else {1:precision@1}
        """
        recalls, precisions = [], []
        for idx in range(0, self.testdata1.shape[0], self.total_num_gt):
            gt_img_id = self.test_img_ids[idx]
            test_datapoint = self.testdata1[idx, :].reshape(1, -1)
            # copy the test text to the number of images
            test_text_emb = np.repeat(test_datapoint, self.testdata2.shape[0], axis=0)
            sim_score = sim_fn(test_text_emb, self.testdata2)
            # sort the similarity score in descending order and get the index
            sim_top_idx = np.argpartition(sim_score, -self.total_num_gt)[
                -self.total_num_gt :
            ]
            sim_top_idx = sim_top_idx[np.argsort(sim_score[sim_top_idx])[::-1]]
            # Recall = Total number of correct data retrieved/Total number of relevant documents in the database
            hit = np.zeros(self.total_num_gt)
            for ii in range(self.total_num_gt):
                hit[ii] = 1 if gt_img_id == self.test_img_ids[sim_top_idx[ii]] else 0
            recall = np.cumsum(hit) / self.total_num_gt
            # Precision = Total number of correct data retrieved/Total number of retrieved documents
            precision = np.cumsum(hit) / (np.arange(self.total_num_gt) + 1)
            recalls.append(recall)
            precisions.append(precision)
        recalls = np.array(recalls).mean(axis=0)
        precisions = np.array(precisions).mean(axis=0)

        if self.total_num_gt == 1:
            return {1: recalls[0]}, {1: precisions[0]}
        elif self.total_num_gt == 5:  # noqa: PLR2004, RET505
            return {1: recalls[0], 5: recalls[4]}, {1: precisions[0], 5: precisions[4]}
        else:
            msg = f"Total number of ground truth {self.total_num_gt} not supported"
            raise ValueError(msg)


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
        super().preprocess_retrieval_data(data1, data2)
        if self.img2text:  # image to retrieve text
            self.data1, self.data2 = data1, data2
            self.total_num_gt = 5  # Total number of relevant texts in the database
        else:  # text to retrieve image
            # data 1 is originally image and data2 is originally text
            self.data1, self.data2 = data2, data1
            self.total_num_gt = 1  # Total number of relevant images in the database

        self.train_idx = np.where(self.splits == "train")[0][:2000]  # 145_000
        self.test_idx = np.where(self.splits == "test")[0][:2000]  # 5_000

        self.traindata1, self.traindata2 = (
            self.data1[self.train_idx],
            self.data2[self.train_idx],
        )
        self.testdata1, self.testdata2 = (
            self.data1[self.test_idx],
            self.data2[self.test_idx],
        )
        self.test_img_ids = self.img_ids[self.test_idx]


def load_retrieval_dataset(cfg: DictConfig) -> BaseRetrievalDataset:
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
