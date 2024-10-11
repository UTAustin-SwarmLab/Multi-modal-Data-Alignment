"""Dataset class for retrieval task."""

from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig

import hydra
from mmda.baselines.asif_core import zero_shot_classification
from mmda.utils.any2any_ds_class import BaseAny2AnyDataset
from mmda.utils.dataset_utils import load_flickr
from mmda.utils.kitti_ds_class import KITTIDataset
from mmda.utils.mstvtt_ds_class import MSRVTTDataset


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
        self.img2txt = None
        self.test_img_ids = None

    def preprocess_retrieval_data(self, data1: np.ndarray, data2: np.ndarray) -> None:
        """Preprocess the data for retrieval. Check the shape of the data.

        Args:
            data1: data from the first encoder
            data2: data from the second encoder
        """
        assert data1.shape[0] == data2.shape[0], f"{data1.shape[0]}!={data2.shape[0]}"

    def map_precision_similarity(
        self, sim_fn: Union[callable, str], cfg: DictConfig = None  # noqa: UP007
    ) -> tuple[float, dict[float:float]]:
        """Calculate the mean average precision and precision at k (1 ~ num_gt).

        Args:
            sim_fn: similarity function
            cfg: configuration file
        Returns:
            map: {mAP}
            precisions: {1: precision@1, 5:precision@5}
            similarity: similarity score
        """
        maps, precisions = [], []
        sim_scores = []
        if sim_fn == "asif":
            # set parameters
            non_zeros = min(cfg.asif.non_zeros, self.traindata1.shape[0])
            range_anch = [
                2**i
                for i in range(
                    int(np.log2(non_zeros) + 1),
                    int(np.log2(len(self.traindata1))) + 2,
                )
            ]
            range_anch = range_anch[-1:]  # run just last anchor to be quick
            val_labels = torch.zeros((1,), dtype=torch.float32)
            _anchors, scores, sim_score_matrix = zero_shot_classification(
                torch.tensor(self.testdata1, dtype=torch.float32),
                torch.tensor(self.testdata2, dtype=torch.float32),
                torch.tensor(self.traindata1, dtype=torch.float32),
                torch.tensor(self.traindata2, dtype=torch.float32),
                val_labels,
                non_zeros,
                range_anch,
                cfg.asif.val_exps,
                max_gpu_mem_gb=cfg.asif.max_gpu_mem_gb,
            )
            sim_score_matrix = sim_score_matrix.numpy().astype(np.float32)
        for idx in range(0, self.testdata1.shape[0], self.num_gt):
            gt_img_id = self.test_img_ids[idx]
            test_datapoint = self.testdata1[idx, :].reshape(1, -1)
            # copy the test text to the number of images
            test_txt_emb = np.repeat(test_datapoint, self.testdata2.shape[0], axis=0)
            sim_score = (
                sim_score_matrix[idx, :]
                if sim_fn == "asif"
                else sim_fn(test_txt_emb, self.testdata2)
            )
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
        sim_scores = np.array(sim_scores)
        return maps, {1: precisions[0], 5: precisions[4]}, sim_scores

    def top_k_presicion(
        self, sim_fn: Union[callable, str], cfg: DictConfig = None  # noqa: UP007
    ) -> tuple[float, dict[float:float]]:
        """Calculate the average precision and precision at k (1 ~ num_gt).

        Args:
            sim_fn: similarity function
            cfg: configuration file
        Returns:
            map: {mAP}
            precisions: {1: precision@1, 5:precision@5}
        """
        return self.map_precision_similarity(sim_fn, cfg)[0:2]


class FlickrDataset(BaseRetrievalDataset):
    """Flickr dataset class."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        super().__init__(cfg)
        self.img_path, self.txt_descriptions, self.splits, self.img_ids = load_flickr(
            cfg["flickr"]
        )
        self.img2txt = cfg["flickr"].img2txt

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
        if self.img2txt:  # image to retrieve text
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


def load_retrieval_dataset(
    cfg: DictConfig,
) -> BaseRetrievalDataset | BaseAny2AnyDataset:
    """Load the dataset for retrieval task.

    Args:
        cfg: configuration file

    Returns:
        dataset: dataset class
    """
    if cfg.dataset == "flickr":
        dataset = FlickrDataset(cfg)
    elif cfg.dataset == "KITTI":
        dataset = KITTIDataset(cfg)
    elif cfg.dataset == "MSRVTT":
        dataset = MSRVTTDataset(cfg)
    else:
        error_message = (
            f"{cfg.dataset} is not supported in {cfg.any_retrieval_datasets}."
        )
        raise ValueError(error_message)

    return dataset


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def test(cfg: DictConfig) -> None:
    """Test the retrieval dataset class.

    Args:
        cfg: configuration file
    """
    ds = KITTIDataset(cfg)
    ds.preprocess_retrieval_data()
    ds.train_crossmodal_similarity()
    ds.generate_cali_data()


if __name__ == "__main__":
    test()
