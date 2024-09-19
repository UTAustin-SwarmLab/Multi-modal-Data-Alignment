"""Dataset class for retrieval task."""

import pickle
from pathlib import Path
from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

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
        self.modal0mask = np.random.choice(self.test_size, mask_num, replace=False)
        self.modal1mask = np.random.choice(self.test_size, mask_num, replace=False)
        self.modal2mask = np.random.choice(self.test_size, mask_num, replace=False)

        self.imgdata["test"][self.modal0mask] = np.nan
        self.lidardata["test"][self.modal1mask] = np.nan
        self.txtdata["test"][self.modal2mask] = np.nan

    def train_crossmodal_similarity(self) -> None:
        """Train the cross-modal similarity, aka the CSA method."""
        cfg_dataset = self.cfg_dataset
        if self.img2lidar == "csa":
            cca_save_path = Path(
                cfg_dataset.paths.save_path
                + "any2any_cca_"
                + f"{cfg_dataset.img_encoder}_{cfg_dataset.lidar_encoder}.pkl"
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
                + "any2any_cca_"
                + f"{cfg_dataset.img_encoder}_{cfg_dataset.text_encoder}.pkl"
            )
            self.img2txt_cca = NormalizedCCA()
            if not cca_save_path.exists():
                self.cca_img2txt, self.cca_txt2img, self.img2txt_corr = (
                    self.img2txt_cca.fit_transform_train_data(
                        self.cfg_dataset, self.imgdata["train"], self.txtdata["train"]
                    )
                )
                self.img2txt_cca.save_model(cca_save_path)
            else:
                self.img2txt_cca.load_model(cca_save_path)
                self.cca_img2txt = self.img2txt_cca.traindata1
                self.cca_txt2img = self.img2txt_cca.traindata2
                self.img2txt_corr = self.img2txt_cca.corr_coeff
        if self.txt2lidar == "csa":
            cca_save_path = Path(
                cfg_dataset.paths.save_path
                + "any2any_cca_"
                + f"{cfg_dataset.text_encoder}_{cfg_dataset.lidar_encoder}.pkl"
            )
            self.txt2lidar_cca = NormalizedCCA()
            if not cca_save_path.exists():
                self.cca_txt2lidar, self.cca_lidar2txt, self.txt2lidar_corr = (
                    self.txt2lidar_cca.fit_transform_train_data(
                        self.cfg_dataset, self.txtdata["train"], self.lidardata["train"]
                    )
                )
                self.txt2lidar_cca.save_model(cca_save_path)
            else:
                self.txt2lidar_cca.load_model(cca_save_path)
                self.cca_txt2lidar = self.txt2lidar_cca.traindata1
                self.cca_lidar2txt = self.txt2lidar_cca.traindata2
                self.txt2lidar_corr = self.txt2lidar_cca.corr_coeff

    def calculate_similarity_matrix(
        self,
        x1: list[list[np.array]],
        x2: list[list[np.array]],
    ) -> np.ndarray:
        """Calculate the similarity matrix.

        Args:
            x1: the first data (masked if possible) shape is (3, 3, emb_dim)
            x2: the second data (masked if possible) shape is (3, 3, emb_dim)

        Returns:
            similarity_matrix: the similarity matrix of a pair of data
        """
        sim_mat = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                csa = False
                x1_, x2_ = x1[i][j], x2[j][i]
                if np.any(np.isnan(x1[i][i])) or np.any(np.isnan(x2[j][j])):
                    sim_mat[i, j] = -1
                elif i == j:
                    csa = False
                elif i + j == 1 and self.img2lidar == "csa":
                    csa = True
                    corr = self.img2lidar_corr
                elif i + j == 2 and self.img2txt == "csa":  # noqa: PLR2004
                    csa = True
                    corr = self.img2txt_corr
                elif i + j == 3 and self.txt2lidar == "csa":  # noqa: PLR2004
                    corr = self.txt2lidar_corr
                    csa = True

                if csa:
                    sim_mat[i, j] = weighted_corr_sim(
                        x=x1_.reshape(1, -1),
                        y=x2_.reshape(1, -1),
                        corr=corr,
                        dim=self.cfg_dataset.retrieval_dim,
                    )[0]
                else:
                    sim_mat[i, j] = cosine_sim(
                        x1[i][j].reshape(1, -1), x2[j][i].reshape(1, -1)
                    )[0]
        return sim_mat

    def transform_with_cca(
        self,
        img_data: list[list[np.array]],
        lidar_data: list[list[np.array]],
        txt_data: list[list[np.array]],
    ) -> list[list[np.array]]:
        """Transform the data with cca.

        Args:
            img_data: the image data
            lidar_data: the lidar data
            txt_data: the text data

        Returns:
            cca_img2lidar: the cca transformed image data to img-lidar space
            cca_lidar2img: the cca transformed lidar data to lidar-img space
            cca_img2txt: the cca transformed image data to img-text space
            cca_txt2img: the cca transformed text data to text-img space
            cca_txt2lidar: the cca transformed text data to text-lidar space
            cca_lidar2txt: the cca transformed lidar data to lidar-text space
        """
        # cca transformation
        if self.img2lidar == "csa":
            cca_img2lidar, cca_lidar2img = self.img2lidar_cca.transform_data(
                img_data, lidar_data
            )
        else:
            cca_img2lidar, cca_lidar2img = (img_data, lidar_data)
        if self.img2txt == "csa":
            cca_img2txt, cca_txt2img = self.img2txt_cca.transform_data(
                img_data, txt_data
            )
        else:
            cca_img2txt, cca_txt2img = (img_data, txt_data)
        if self.txt2lidar == "csa":
            cca_txt2lidar, cca_lidar2txt = self.txt2lidar_cca.transform_data(
                txt_data, lidar_data
            )
        else:
            cca_txt2lidar, cca_lidar2txt = (txt_data, lidar_data)
        return (
            cca_img2lidar,
            cca_lidar2img,
            cca_img2txt,
            cca_txt2img,
            cca_txt2lidar,
            cca_lidar2txt,
        )

    def calculate_pairs_data_similarity(
        self,
        img_data: np.array,
        lidar_data: np.array,
        txt_data: np.array,
        idx_offset: int,
    ) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
        """Calculate the similarity matrices of all pairs of data, given in the args.

        Args:
            img_data: the image data
            lidar_data: the lidar data
            txt_data: the text data
            idx_offset: the index offset (calibration = train_size + test_size, test = train_size)

        Returns:
            sim_mat: the similarity matrices of a pair of data.
                key is the pair of indices, value is the similarity matrix and ground truth label.
        """
        (
            cca_img2lidar,
            cca_lidar2img,
            cca_img2txt,
            cca_txt2img,
            cca_txt2lidar,
            cca_lidar2txt,
        ) = self.transform_with_cca(img_data, lidar_data, txt_data)
        ds_size = img_data.shape[0]
        # calculate the similarity matrix, we do not mask the data here
        sim_mat_cali = {}  # (i, j) -> (sim_mat, gt_label)
        for i in tqdm(range(ds_size), desc="Calibrate cross-modal similarity i"):
            for j in tqdm(range(i, ds_size), desc="Calibrate cross-modal similarity j"):
                ds_idx_q = self.shuffle2idx[i + idx_offset]
                ds_idx_r = self.shuffle2idx[j + idx_offset]
                gt_label = eval_retrieval_ids(ds_idx_q, ds_idx_r)
                # if gt_label == 0:  # only consider the negative pairs
                sim_mat = self.calculate_similarity_matrix(
                    x1=[
                        [
                            img_data[i],
                            cca_img2lidar[i],
                            cca_img2txt[i],
                        ],
                        [
                            cca_lidar2img[i],
                            lidar_data[i],
                            cca_lidar2txt[i],
                        ],
                        [
                            cca_txt2img[i],
                            cca_txt2lidar[i],
                            txt_data[i],
                        ],
                    ],
                    x2=[
                        [
                            img_data[j],
                            cca_img2lidar[j],
                            cca_img2txt[j],
                        ],
                        [
                            cca_lidar2img[j],
                            lidar_data[j],
                            cca_lidar2txt[j],
                        ],
                        [
                            cca_txt2img[j],
                            cca_txt2lidar[j],
                            txt_data[j],
                        ],
                    ],
                )
                sim_mat_cali[(ds_idx_q, ds_idx_r)] = (sim_mat, gt_label)
        return sim_mat_cali

    def calibrate_crossmodal_similarity(self) -> None:
        """Calibrate the cross-modal similarity. Save the similarity matrix in the format of (sim_score, gt_label)."""
        img_data = self.imgdata["cali"]
        lidar_data = self.lidardata["cali"]
        txt_data = self.txtdata["cali"]
        idx_offset = self.train_size + self.test_size
        sim_mat_cali = self.calculate_pairs_data_similarity(
            img_data, lidar_data, txt_data, idx_offset
        )
        # save the calibration data in the format of (sim_score, gt_label)
        with Path(
            self.cfg_dataset.paths.save_path,
            f"sim_mat_cali_{self.cfg_dataset.retrieval_dim}.pkl",
        ).open("wb") as f:
            pickle.dump(sim_mat_cali, f)

    def retrieve_data(self) -> np.ndarray:
        """Retrieve the data for retrieval task on the test set.

        Returns:
            similarity_matrix: the similarity matrix of a pair of data
        """
        # mask the data (the original test data is masked, but not the cca transformed data)
        # cca transformation
        if self.img2lidar == "csa":
            self.cca_img2lidar_test, self.cca_lidar2img_test = (
                self.img2lidar_cca.transform_data(
                    self.imgdata["test"], self.lidardata["test"]
                )
            )
        if self.img2txt == "csa":
            self.cca_img2txt_test, self.cca_txt2img_test = (
                self.img2txt_cca.transform_data(
                    self.imgdata["test"], self.txtdata["test"]
                )
            )
        if self.txt2lidar == "csa":
            self.cca_txt2lidar_test, self.cca_lidar2txt_test = (
                self.txt2lidar_cca.transform_data(
                    self.txtdata["test"], self.lidardata["test"]
                )
            )


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
        error_message = (
            f"{cfg.dataset} is not supported. Try {cfg.any_retrieval_datasets}."
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
    ds.calibrate_crossmodal_similarity()


if __name__ == "__main__":
    test()
