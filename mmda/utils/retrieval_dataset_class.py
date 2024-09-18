"""Dataset class for retrieval task."""

import pickle
from pathlib import Path
from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig

import hydra
from mmda.baselines.asif_core import zero_shot_classification
from mmda.utils.cca_class import NormalizedCCA
from mmda.utils.data_utils import load_three_encoder_data
from mmda.utils.dataset_utils import load_flickr
from mmda.utils.liploc_model import eval_retrieval_ids
from mmda.utils.sim_utils import cosine_sim, weighted_corr_sim


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


class KITTIDataset:
    """KITTI dataset class."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        np.random.seed(0)
        self.cfg = cfg

        self.img2img = cfg["KITTI"].img_encoder
        self.lidar2lidar = cfg["KITTI"].lidar_encoder
        self.txt2txt = cfg["KITTI"].text_encoder
        self.img2lidar = cfg["KITTI"].lidar_encoder
        self.img2txt = "csa"
        self.txt2lidar = "csa"

        self.cali_size = 97
        self.train_size = 5000

    def preprocess_retrieval_data(self) -> None:
        """Preprocess the data for retrieval."""
        # load data
        self.cfg_dataset, imgdata, lidardata, txtdata = load_three_encoder_data(
            self.cfg
        )
        self.num_data = imgdata.shape[0]
        self.test_size = self.num_data - self.cali_size - self.train_size
        assert (
            self.num_data == lidardata.shape[0]
        ), f"{self.num_data}!={lidardata.shape[0]}"
        assert self.num_data == txtdata.shape[0], f"{self.num_data}!={txtdata.shape[0]}"

        # train/test/calibration split
        idx = np.arange(self.num_data)  # An array with 100 elements
        # Shuffle the array to ensure randomness
        np.random.shuffle(idx)
        self.idx2shuffle = {i: idx[i] for i in range(self.num_data)}
        self.shuffle2idx = {idx[i]: i for i in range(self.num_data)}
        self.train_idx = idx[: self.train_size]
        self.test_idx = idx[self.train_size : -self.cali_size]
        self.cali_idx_qdx = idx[-self.cali_size :]
        print(
            f"train: {self.train_idx.shape}, test: {self.test_idx.shape}, cali: {self.cali_idx_qdx.shape}"
        )
        self.imgdata = {
            "train": imgdata[self.train_idx],
            "test": imgdata[self.test_idx],
            "cali": imgdata[self.cali_idx_qdx],
        }
        self.lidardata = {
            "train": lidardata[self.train_idx],
            "test": lidardata[self.test_idx],
            "cali": lidardata[self.cali_idx_qdx],
        }
        self.txtdata = {
            "train": txtdata[self.train_idx],
            "test": txtdata[self.test_idx],
            "cali": txtdata[self.cali_idx_qdx],
        }

        # masking missing data in the test set. Mask the whole modality of an instance at a time.
        mask_num = int(self.test_size / 2)
        self.modal1mask = np.random.choice(self.test_size, mask_num, replace=False)
        self.modal2mask = np.random.choice(self.test_size, mask_num, replace=False)
        self.modal3mask = np.random.choice(self.test_size, mask_num, replace=False)

        self.imgdata["test"][self.modal1mask] = np.nan
        self.lidardata["test"][self.modal2mask] = np.nan
        self.txtdata["test"][self.modal3mask] = np.nan

    def train_crossmodal_similarity(self) -> None:
        """Train the cross-modal similarity, aka the CSA method."""
        cfg_dataset = self.cfg_dataset
        if self.img2lidar == "csa":
            cca_save_path = Path(
                cfg_dataset.paths.save_path
                + "any2any_cca"
                + f"{cfg_dataset.img_encoder}_{cfg_dataset.lidar_encoder}_{cfg_dataset.sim_dim}.pkl"
            )
            self.img2lidar_cca = NormalizedCCA()
            if not cca_save_path.exists():
                self.cca_img2lidar, self.cca_lidar2img, self.img2lidar_corr = (
                    self.img2lidar_cca.fit_transform_train_data(
                        self.cfg_dataset, self.imgdata["train"], self.lidardata["train"]
                    )
                )
                self.img2lidar_cca.save_model(cca_save_path)
            else:
                self.img2lidar_cca.load_model(cca_save_path)
                self.cca_img2lidar = self.img2lidar_cca.traindata1
                self.cca_lidar2img = self.img2lidar_cca.traindata2
                self.img2lidar_corr = self.img2lidar_cca.corr_coeff
        if self.img2txt == "csa":
            cca_save_path = Path(
                cfg_dataset.paths.save_path
                + "any2any_cca"
                + f"{cfg_dataset.img_encoder}_{cfg_dataset.text_encoder}_{cfg_dataset.sim_dim}.pkl"
            )
            self.img2txt_cca = NormalizedCCA()
            if not cca_save_path.exists():
                self.cca_img2text, self.cca_txt2img, self.img2txt_corr = (
                    self.img2txt_cca.fit_transform_train_data(
                        self.cfg_dataset, self.imgdata["train"], self.txtdata["train"]
                    )
                )
                self.img2txt_cca.save_model(cca_save_path)
            else:
                self.img2txt_cca.load_model(cca_save_path)
                self.cca_img2text = self.img2txt_cca.traindata1
                self.cca_txt2img = self.img2txt_cca.traindata2
                self.img2txt_corr = self.img2txt_cca.corr_coeff
        if self.txt2lidar == "csa":
            cca_save_path = Path(
                cfg_dataset.paths.save_path
                + "any2any_cca"
                + f"{cfg_dataset.text_encoder}_{cfg_dataset.lidar_encoder}_{cfg_dataset.sim_dim}.pkl"
            )
            self.txt2lidar_cca = NormalizedCCA()
            if not cca_save_path.exists():
                self.cca_txt2lidar, self.cca_lidar2text, self.txt2lidar_corr = (
                    self.txt2lidar_cca.fit_transform_train_data(
                        self.cfg_dataset, self.txtdata["train"], self.lidardata["train"]
                    )
                )
                self.txt2lidar_cca.save_model(cca_save_path)
            else:
                self.txt2lidar_cca.load_model(cca_save_path)
                self.cca_txt2lidar = self.txt2lidar_cca.traindata1
                self.cca_lidar2text = self.txt2lidar_cca.traindata2
                self.txt2lidar_corr = self.txt2lidar_cca.corr_coeff

    def calculate_similarity_matrix(  # noqa: PLR0912, C901
        self,
        x1: tuple[np.ndarray, np.ndarray, np.ndarray],
        x2: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Calculate the similarity matrix.

        Args:
            x1: the first data (image, lidar, text) (masked if possible)
            x2: the second data (image, lidar, text) (masked if possible)

        Returns:
            similarity_matrix: the similarity matrix of a pair of data
        """
        sim_mat = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                csa = False
                if np.any(np.isnan(x1[i])) or np.any(np.isnan(x2[j])):
                    sim_mat[i, j] = -1
                elif i == j:
                    csa = False
                elif i + j == 1 and self.img2lidar == "csa":
                    corr = self.img2lidar_corr
                    if i == 1 and j == 0:  # img and lidar
                        x1_, x2_ = x1[i].copy(), x2[j].copy()
                    else:  # lidar and img
                        x1_, x2_ = x2[j].copy(), x1[i].copy()
                    x1_, x2_ = self.img2lidar_cca.transform_data(
                        x1_.reshape(1, -1), x2_.reshape(1, -1)
                    )
                    csa = True
                elif i + j == 2 and self.img2txt == "csa":  # noqa: PLR2004
                    corr = self.img2txt_corr
                    if i == 0 and j == 2:  # img and text # noqa: PLR2004
                        x1_, x2_ = x1[i].copy(), x2[j].copy()
                    else:  # text and img
                        x1_, x2_ = x2[j].copy(), x1[i].copy()
                    x1_, x2_ = self.img2txt_cca.transform_data(
                        x1_.reshape(1, -1), x2_.reshape(1, -1)
                    )
                    csa = True
                elif i + j == 3 and self.txt2lidar == "csa":  # noqa: PLR2004
                    corr = self.txt2lidar_corr
                    if i == 2 and j == 1:  # text and lidar  # noqa: PLR2004
                        x1_, x2_ = x1[i].copy(), x2[j].copy()
                    else:  # lidar and text
                        x1_, x2_ = x2[j].copy(), x1[i].copy()
                    x1_, x2_ = self.txt2lidar_cca.transform_data(
                        x1_.reshape(1, -1), x2_.reshape(1, -1)
                    )
                    csa = True

                if csa:
                    sim_mat[i, j] = weighted_corr_sim(
                        x=x1_,
                        y=x2_,
                        corr=corr,
                        dim=self.cfg_dataset.sim_dim,
                    )[0]
                else:
                    sim_mat[i, j] = cosine_sim(
                        x1[i].reshape(1, -1), x2[j].reshape(1, -1)
                    )[0]
        return sim_mat

    def calibrate_crossmodal_similarity(self) -> None:
        """Calibrate the cross-modal similarity."""
        # cca transformation
        if self.img2lidar == "csa":
            self.cca_img2lidar_cali, self.cca_lidar2img_cali = (
                self.img2lidar_cca.transform_data(
                    self.imgdata["cali"], self.lidardata["cali"]
                )
            )
        if self.img2txt == "csa":
            self.cca_img2text_cali, self.cca_txt2img_cali = (
                self.img2txt_cca.transform_data(
                    self.imgdata["cali"], self.txtdata["cali"]
                )
            )
        if self.txt2lidar == "csa":
            self.cca_txt2lidar_cali, self.cca_lidar2text_cali = (
                self.txt2lidar_cca.transform_data(
                    self.txtdata["cali"], self.lidardata["cali"]
                )
            )
        # calculate the similarity matrix, we do not mask the data here
        sim_mat_cali = {}  # (i, j) -> (sim_mat, gt_label)
        for cali_q in range(self.cali_size):
            for cali_r in range(cali_q, self.cali_size):
                ds_idx_q = self.shuffle2idx[cali_q + self.train_size + self.test_size]
                ds_idx_r = self.shuffle2idx[cali_r + self.train_size + self.test_size]
                gt_label = eval_retrieval_ids(ds_idx_q, ds_idx_r)
                if gt_label == 0:  # only consider the negative pairs
                    sim_mat = self.calculate_similarity_matrix(
                        (
                            self.imgdata["cali"][cali_q],
                            self.lidardata["cali"][cali_q],
                            self.txtdata["cali"][cali_q],
                        ),
                        (
                            self.imgdata["cali"][cali_r],
                            self.lidardata["cali"][cali_r],
                            self.txtdata["cali"][cali_r],
                        ),
                    )
                    sim_mat_cali[(ds_idx_q, ds_idx_r)] = (sim_mat, gt_label)

        # save the calibration data in the format of (sim_score, gt_label)
        with (
            Path(self.cfg_dataset.paths.save_path) / "sim_mat_cali.pkl".open("wb") as f
        ):
            pickle.dump(sim_mat_cali, f)


def load_retrieval_dataset(cfg: DictConfig) -> FlickrDataset | KITTIDataset:
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
    else:
        msg = f"Dataset {cfg.dataset} not supported"
        raise ValueError(msg)
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
    ds.calibrate_crossmodal_similarity()


if __name__ == "__main__":
    test()
