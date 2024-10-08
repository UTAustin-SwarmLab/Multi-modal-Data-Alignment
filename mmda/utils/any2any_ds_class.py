"""Dataset class for any2any retrieval task."""

import copy
import pickle
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from mmda.utils.calibrate import calibrate, get_non_conformity_scores_1st_stage
from mmda.utils.cca_class import NormalizedCCA
from mmda.utils.data_utils import load_three_encoder_data
from mmda.utils.dataset_utils import load_msrvtt
from mmda.utils.liploc_model import KITTI_file_Retrieval, get_top_k
from mmda.utils.sim_utils import batch_weighted_corr_sim, cosine_sim


class BaseAny2AnyDataset:
    """Base class for any2any retrieval dataset."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset."""

    def preprocess_retrieval_data(self) -> None:
        """Preprocess the data for retrieval."""

    def train_crossmodal_similarity(self) -> None:
        """Train the cross-modal similarity, aka the CSA method."""

    def get_cali_data(self) -> None:
        """Get the calibration data."""

    def set_pred_band(self, ij_range: tuple[int, int] = (3, 3)) -> None:
        """Set up the prediction bands for the calibration."""
        self.nc_scores = {}
        print("Calculating nonconformity scores...")
        # calculate the nonconformity scores and conformal scores for all pairs of modalities
        for i in range(ij_range[0]):
            for j in range(ij_range[1]):
                self.nc_scores[(i, j)] = get_non_conformity_scores_1st_stage(
                    self.sim_mat_cali, i, j
                )[0]

    def get_test_data(self, data_lists: list[np.ndarray]) -> None:
        """Get the test data. Create the similarity matrix in the format of (sim_score, gt_label).

        This step is extremely time-consuming, so we cache the similarity matrix in the pickle format
        and use batch processing to speed up the process.

        Args:
            data_lists: list of data
        """
        if Path(
            self.cfg_dataset.paths.save_path,
            f"con_mat_test_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}.pkl",
        ).exists():
            print(
                "Since the conformal probabilities are already calculated, we skip the process of loading test data."
            )
            return

        sim_mat_test_path = Path(
            self.cfg_dataset.paths.save_path,
            f"sim_mat_test_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}.pkl",
        )
        if not sim_mat_test_path.exists():
            print("Generating test data...")
            idx_offset = self.train_size
            self.sim_mat_test = self.calculate_pairs_data_similarity(
                data_lists,
                idx_offset,
                num_workers=8,
            )
            with sim_mat_test_path.open("wb") as f:
                pickle.dump(self.sim_mat_test, f)
        else:
            print("Loading test data...")
            # load with pickle since it is faster than joblib (but less safe)
            with sim_mat_test_path.open("rb") as f:
                self.sim_mat_test = pickle.load(f)  # noqa: S301

    def cal_test_conformal_prob(self) -> None:
        """Calculate the conformal probability for the test data.

        Args:
            shape: the shape of the similarity matrix
        """
        con_mat_test_path = Path(
            self.cfg_dataset.paths.save_path,
            f"con_mat_test_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}.pkl",
        )
        if not con_mat_test_path.exists():
            shape = self.shape
            self.con_mat_test = {}
            for (idx_q, idx_r), (sim_mat, gt_label) in tqdm(
                self.sim_mat_test.items(),
                desc="Calculating conformal probabilities",
                leave=True,
            ):
                probs = np.zeros(shape)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        probs[i, j] = calibrate(sim_mat[i, j], self.nc_scores[(i, j)])
                self.con_mat_test[(idx_q, idx_r)] = (probs, gt_label)
            with con_mat_test_path.open("wb") as f:
                pickle.dump(self.con_mat_test, f)
        else:
            print("Loading conformal probabilities...")
            # load with pickle since it is faster than joblib (but less safe)
            with con_mat_test_path.open("rb") as f:
                self.con_mat_test = pickle.load(f)  # noqa: S301

        con_mat_test_miss_path = Path(
            self.cfg_dataset.paths.save_path,
            f"con_mat_test_miss_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}.pkl",
        )
        if not con_mat_test_miss_path.exists():
            self.con_mat_test_miss = copy.deepcopy(self.con_mat_test)
            shape = self.shape
            for (idx_q, idx_r), (_, _) in tqdm(
                self.con_mat_test.items(),
                desc="Calculating conformal probabilities for missing data",
                leave=True,
            ):
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        if idx_q in self.mask[i] and idx_r in self.mask[j]:
                            self.con_mat_test_miss[(idx_q, idx_r)][0][i, j] = -1
            with con_mat_test_miss_path.open("wb") as f:
                pickle.dump(self.con_mat_test_miss, f)
        else:
            print("Loading conformal probabilities for missing data...")
            # load with pickle since it is faster than joblib (but less safe)
            with con_mat_test_miss_path.open("rb") as f:
                self.con_mat_test_miss = pickle.load(f)  # noqa: S301

    def parse_retrieved_pairs(
        self,
        ds_idx_q: int,
        ds_idx_r: int,
        con_mat: dict,
        single_modal: bool = False,
        select_top_k: int = 9,
    ) -> tuple[int, int, float, int]:
        """Parse the retrieved pairs.

        Args:
            ds_idx_q: the index of the query data
            ds_idx_r: the index of the retrieved data
            con_mat: the conformal probability matrix
            single_modal: whether to retrieve the single modality data
            select_top_k: the number of top-k conformal probabilities to use for the mean

        Returns:
            ds_idx_q: the index of the query data
            ds_idx_r: the index of the retrieved data
            conformal_prob: the conformal probability (sometimes the mean of the top-k conformal probabilities)
            gt_label: the ground truth label
        """
        probs = con_mat[(ds_idx_q, ds_idx_r)][0]
        if single_modal:  # single modality retrieval
            return (ds_idx_q, ds_idx_r, probs, con_mat[(ds_idx_q, ds_idx_r)][1])
        # full modality retrieval
        probs_filtered = probs[probs != -1]  # Ignore -1 entries
        if len(probs_filtered) > 0:
            probs_sorted = np.sort(probs_filtered)
            top_k_mean = (
                np.mean(probs_sorted[-select_top_k:])
                if len(probs_sorted) >= select_top_k
                else probs_sorted[-1]
            )
        else:
            top_k_mean = 0  # Default value if all entries are -1
        return (
            ds_idx_q,
            ds_idx_r,
            top_k_mean,  # conformal_prob
            con_mat[(ds_idx_q, ds_idx_r)][1],  # gt_label
        )

    def retrieve_data(
        self,
        mode: Literal["miss", "full", "single"],
    ) -> tuple[dict, dict, dict]:
        """Retrieve the data for retrieval task on the test set.

        Args:
            mode: the mode of the retrieval. "miss" for the retrieval on the missing data,
                "full" for the retrieval on the full data, "single" for the retrieval on single pair of modalities.

        Returns:
            recalls: dict of the recall at 1, 5, 20. {int: list}
            precisions: dict of the precision at 1, 5, 20. {int: list}
            maps: dict of the mean average precision at 5, 20. {int: list}
        """
        assert mode in [
            "miss",
            "full",
            "single",
        ], f"mode must be 'miss' or 'full' or 'single', got {mode}"

        con_mat = self.con_mat_test_miss if mode == "miss" else self.con_mat_test

        if mode != "single":
            recalls = {1: [], 5: [], 20: []}
            precisions = {1: [], 5: [], 20: []}
            maps = {5: [], 20: []}

            for idx_q in tqdm(
                range(self.test_size),
                desc=f"Retrieving {mode} data",
                leave=True,
            ):
                retrieved_pairs = self.retrieve_one_data(
                    con_mat, idx_q, self.train_size, self.test_size
                )
                retrieved_pairs.sort(key=lambda x: x[2], reverse=True)
                top_1_hit = get_top_k(retrieved_pairs, k=1)
                top_5_hit = get_top_k(retrieved_pairs, k=5)
                top_20_hit = get_top_k(retrieved_pairs, k=20)

                # calculate recall@1, recall@5, recall@20
                recall_1 = 1 if any(top_1_hit) else 0
                recall_5 = 1 if any(top_5_hit) else 0
                recall_20 = 1 if any(top_20_hit) else 0

                # calculate precision@1, precision@5, precision@20
                precision_1 = sum(top_1_hit) / len(top_1_hit)
                precision_5 = sum(top_5_hit) / len(top_5_hit)
                precision_20 = sum(top_20_hit) / len(top_20_hit)

                # calculate AP@5, AP@20
                precisions_at_5 = np.cumsum(top_5_hit) / (np.arange(5) + 1)  # array
                ap_5 = np.sum(precisions_at_5 * top_5_hit) / 5
                precisions_at_20 = np.cumsum(top_20_hit) / (np.arange(20) + 1)  # array
                ap_20 = np.sum(precisions_at_20 * top_20_hit) / 20

                # record the results
                recalls[1].append(recall_1)
                recalls[5].append(recall_5)
                recalls[20].append(recall_20)
                precisions[1].append(precision_1)
                precisions[5].append(precision_5)
                precisions[20].append(precision_20)
                maps[5].append(ap_5)
                maps[20].append(ap_20)
            return maps, precisions, recalls

        # if mode is "single"
        recalls = {(i, j): [] for i in range(3) for j in range(3)}
        precisions = {(i, j): [] for i in range(3) for j in range(3)}
        maps = {(i, j): [] for i in range(3) for j in range(3)}
        for idx_q in tqdm(
            range(self.test_size),
            desc=f"Retrieving {mode} data",
            leave=True,
        ):
            retrieved_pairs = self.retrieve_one_data(
                con_mat, idx_q, self.train_size, self.test_size, single_modal=True
            )
            for i in range(3):
                for j in range(3):
                    # sort the retrieved pairs not inplace
                    modal_pair = (i, j)
                    retrieved_pairs_ij = sorted(
                        retrieved_pairs, key=lambda x: x[2][modal_pair], reverse=True
                    )
                    top_k_hit = get_top_k(retrieved_pairs_ij, k=5)
                    recall_k = 1 if any(top_k_hit) else 0
                    precision_k = sum(top_k_hit) / len(top_k_hit)
                    recalls[modal_pair].append(recall_k)
                    precisions[modal_pair].append(precision_k)
                    maps[modal_pair].append(precision_k)

        for modal_pair in recalls:
            recalls[modal_pair] = np.mean(recalls[modal_pair])
            precisions[modal_pair] = np.mean(precisions[modal_pair])
            maps[modal_pair] = np.mean(maps[modal_pair])
        return maps, precisions, recalls


class MSRVTTDataset(BaseAny2AnyDataset):
    """MSRVTT dataset class for any2any retrieval task."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset. Size of the data is 52700 in total.

        In this task, we only consider text-to-image/audio retrieval.

        Args:
            cfg: configuration file
        """
        self.cfg = cfg
        self.cfg_dataset = cfg["MSRVTT"]
        self.text2img = "clip"
        self.text2audio = "clap"

        self.shape = (1, 2)  # shape of the similarity matrix
        self.cali_size = 2700
        self.train_size = 0  # no training data is needed for MSRVTT
        self.test_size = 50_000
        self.step_size = 20

    def load_data(self) -> None:
        """Load the data for retrieval."""
        # load data
        self.sen_ids, self.captions, self.video_info_sen_order, self.video_dict = (
            load_msrvtt(self.cfg_dataset)
        )
        self.txt2img_emb = joblib.load(
            Path(self.cfg_dataset.paths.save_path + "MSRVTT_text_emb_clip.pkl")
        )
        self.img2txt_emb = joblib.load(
            Path(self.cfg_dataset.paths.save_path + "MSRVTT_video_emb_clip.pkl")
        )
        self.txt2audio_emb = joblib.load(
            Path(self.cfg_dataset.paths.save_path + "MSRVTT_text_emb_clap.pkl")
        )
        self.audio2txt_emb = joblib.load(
            Path(self.cfg_dataset.paths.save_path + "MSRVTT_audio_emb_clap.pkl")
        )
        # normalize all the embeddings to have unit norm using L2 normalization
        self.txt2img_emb = self.txt2img_emb / np.linalg.norm(
            self.txt2img_emb, axis=1, keepdims=True
        )
        self.img2txt_emb[:, : int(self.img2txt_emb.shape[1] / 2)] = self.img2txt_emb[
            :, : int(self.img2txt_emb.shape[1] / 2)
        ] / np.linalg.norm(
            self.img2txt_emb[:, : int(self.img2txt_emb.shape[1] / 2)],
            axis=1,
            keepdims=True,
        )
        self.img2txt_emb[:, int(self.img2txt_emb.shape[1] / 2) :] = self.img2txt_emb[
            :, int(self.img2txt_emb.shape[1] / 2) :
        ] / np.linalg.norm(
            self.img2txt_emb[:, int(self.img2txt_emb.shape[1] / 2) :],
            axis=1,
            keepdims=True,
        )
        self.txt2audio_emb = self.txt2audio_emb / np.linalg.norm(
            self.txt2audio_emb, axis=1, keepdims=True
        )
        self.audio2txt_emb = self.audio2txt_emb / np.linalg.norm(
            self.audio2txt_emb, axis=1, keepdims=True
        )
        # get the avg of video embeddings, as it does not affect the cosine similarity
        self.img2txt_emb = (
            self.img2txt_emb[:, : int(self.img2txt_emb.shape[1] / 2)]
            + self.img2txt_emb[:, int(self.img2txt_emb.shape[1] / 2) :]
        ) / 2

    def preprocess_retrieval_data(self) -> None:
        """Preprocess the data for retrieval."""
        # load data
        self.load_data()
        self.num_data = self.img2txt_emb.shape[0]
        assert (
            self.test_size + self.cali_size + self.train_size == self.num_data
        ), f"{self.test_size} + {self.cali_size} + {self.train_size} != {self.num_data}"

        # train/test/calibration split
        idx = np.arange(self.num_data)  # An array with 100 elements
        # Shuffle the array to ensure randomness
        np.random.shuffle(idx)
        self.test_idx = idx[self.train_size : -self.cali_size]
        self.cali_idx_qdx = idx[-self.cali_size :]
        self.txt2img_emb = {
            "test": self.txt2img_emb[self.test_idx],
            "cali": self.txt2img_emb[self.cali_idx_qdx],
        }
        self.img2txt_emb = {
            "test": self.img2txt_emb[self.test_idx],
            "cali": self.img2txt_emb[self.cali_idx_qdx],
        }
        self.txt2audio_emb = {
            "test": self.txt2audio_emb[self.test_idx],
            "cali": self.txt2audio_emb[self.cali_idx_qdx],
        }
        self.audio2txt_emb = {
            "test": self.audio2txt_emb[self.test_idx],
            "cali": self.audio2txt_emb[self.cali_idx_qdx],
        }
        # masking missing data in the test set. Mask the whole modality of an instance at a time.
        mask_num = int(self.test_size / self.cfg_dataset.mask_ratio)
        self.mask = {}  # modality -> masked idx
        self.mask[0] = np.random.choice(self.test_size, mask_num, replace=False)
        self.mask[1] = np.random.choice(self.test_size, mask_num, replace=False)

    def check_correct_retrieval(self, q_idx: int, r_idx: int) -> bool:
        """Check if the retrieval is correct.

        Args:
            q_idx: the query index
            r_idx: the retrieved index

        Returns:
            True if the retrieval is correct, False otherwise
        """
        return (
            self.video_info_sen_order[q_idx]["video_id"]
            == self.video_info_sen_order[r_idx]["video_id"]
        )

    def calculate_pairs_data_similarity(
        self,
        data_lists: list[np.ndarray],
        idx_offset: int,
        num_workers: int = 8,
    ) -> np.ndarray:
        """Calculate the similarity matrix for the pairs of modalities.

        Args:
            data_lists: list of data
            idx_offset: the index offset (calibration = train_size + test_size, test = train_size)
            num_workers: the number of workers to use

        Returns:
            sim_mat: the similarity matrices of a pair of data.
                key is the pair of indices in the original dataset,
                value is the similarity matrix (num_data, 1, 2) and ground truth label.
        """
        print(
            f"Calculating similarity matrix... num_workers={num_workers} is not used here."
        )
        ds_size = data_lists[0].shape[0]
        (txt2img_data, img2txt_data, txt2audio_data, audio2txt_data) = data_lists
        ds_size = txt2img_data.shape[0]
        sim_mat = {}
        for i in tqdm(range(ds_size)):
            for j in range(0, ds_size, self.step_size):
                gt_label = self.check_correct_retrieval(i + idx_offset, j + idx_offset)
                cosine_sim_txt2img = np.sum(txt2img_data[i] * img2txt_data[j])
                cosine_sim_txt2audio = np.sum(txt2audio_data[i] * audio2txt_data[j])
                sim_mat[(i + idx_offset, j + idx_offset)] = (
                    np.array([cosine_sim_txt2img, cosine_sim_txt2audio]).reshape(1, -1),
                    gt_label,
                )
        return sim_mat

    def get_cali_data(self) -> None:
        """Get the calibration data.

        Calculate and save the similarity matrix in the format of (sim_score, gt_label).
        Then, we run the calibration to get the conformal scores and obtain the prediction bands.
        """
        sim_mat_path = Path(
            self.cfg_dataset.paths.save_path,
            f"sim_mat_cali_{self.cfg_dataset.mask_ratio}.pkl",
        )
        if not sim_mat_path.exists():
            print("Generating calibration data...")
            txt2img_data = self.txt2img_emb["cali"]
            img2txt_data = self.img2txt_emb["cali"]
            txt2audio_data = self.txt2audio_emb["cali"]
            audio2txt_data = self.audio2txt_emb["cali"]
            idx_offset = self.train_size + self.test_size
            self.sim_mat_cali = self.calculate_pairs_data_similarity(
                (txt2img_data, img2txt_data, txt2audio_data, audio2txt_data),
                idx_offset,
            )
            # save the calibration data in the format of (sim_score, gt_label)
            with sim_mat_path.open("wb") as f:
                pickle.dump(self.sim_mat_cali, f)
        else:
            print("Loading calibration data...")
            self.sim_mat_cali = joblib.load(sim_mat_path.open("rb"))

        # set up prediction bands
        self.set_pred_band((1, 2))

    def get_test_data(self) -> None:
        """Get the test data. Create the similarity matrix in the format of (sim_score, gt_label).

        This step is extremely time-consuming, so we cache the similarity matrix in the pickle format
        and use batch processing to speed up the process.
        """
        super().get_test_data(
            (
                self.img2txt_emb["test"],
                self.txt2img_emb["test"],
                self.txt2audio_emb["test"],
                self.audio2txt_emb["test"],
            )
        )

    def retrieve_one_data(
        self,
        con_mat: dict[tuple[int, int], tuple[np.ndarray, int]],
        idx_q: int,
        idx_offset: int,
        range_r: int,
        single_modal: bool = False,
    ) -> np.ndarray:
        """Retrieve one data from the similarity matrix.

        Args:
            con_mat: the conformal probability matrix.
            idx_q: the index of the query data
            idx_offset: the index offset (calibration = train_size + test_size, test = train_size)
            range_r: the range of the indices to retrieve. (test: (0, test_size), cali: (0, cali_size))
            single_modal: whether to retrieve the single modality data.

        Returns:
            retrieved_pairs: the retrieved pairs in the format of (modal_idx_1, modal_idx_2, conformal_prob, gt_label)
                and in descending order of the conformal probability.
        """
        retrieved_pairs = []
        ds_idx_q = idx_q + idx_offset
        for idx_r in range(0, range_r, self.step_size):
            if idx_r == idx_q:  # cannot retrieve itself
                continue
            ds_idx_r = idx_r + idx_offset
            retrieved_pairs.append(
                self.parse_retrieved_pairs(ds_idx_q, ds_idx_r, con_mat, single_modal, 2)
            )
        return retrieved_pairs


class KITTIDataset(BaseAny2AnyDataset):
    """KITTI dataset class for any2any retrieval task."""

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
        # total 12097
        self.cali_size = 1097
        self.train_size = 5000
        self.shape = (3, 3)  # shape of the similarity matrix
        self.shuffle_step = cfg["KITTI"].shuffle_step

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
        if self.shuffle_step == 0:
            idx = np.arange(self.num_data)  # An array with 100 elements
            # Shuffle the array to ensure randomness
            np.random.shuffle(idx)
        else:
            idx_step = np.arange(0, self.num_data, self.shuffle_step)
            # Shuffle the array to ensure randomness
            np.random.shuffle(idx_step)
            idx = []
            for id_step in idx_step:
                for j in range(self.shuffle_step):
                    if j + id_step < self.num_data:
                        idx.append(j + id_step)
            idx = np.array(idx)
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
        mask_num = int(self.test_size / self.cfg_dataset.mask_ratio)
        self.mask = {}  # modality -> masked idx
        self.mask[0] = np.random.choice(self.test_size, mask_num, replace=False)
        self.mask[1] = np.random.choice(self.test_size, mask_num, replace=False)
        self.mask[2] = np.random.choice(self.test_size, mask_num, replace=False)

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
            x1: the first data (not masked) shape is [3, 3], np: num_data, emb_dim)
            x2: the second data (not masked) shape is [3, 3], np: num_data, emb_dim)

        Returns:
            similarity_matrix: the similarity matrix of a pair of data shape is (num_data, 3, 3)
        """
        num_data = x1[0][0].shape[0]
        sim_mat = np.zeros((num_data, 3, 3))
        for i in range(3):
            for j in range(3):
                csa = False
                x1_ = x1[i][j]
                x2_ = x2[i][j]
                if np.any(np.isnan(x1_)) or np.any(np.isnan(x2_)):
                    sim_mat[:, i, j] = -1
                    msg = "NaN in the data, did you mask the data?"
                    raise ValueError(msg)
                if i == j:
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
                    sim_mat[:, i, j] = batch_weighted_corr_sim(
                        x=x1_,
                        y=x2_,
                        corr=corr,
                        dim=self.cfg_dataset.retrieval_dim,
                    )
                else:
                    sim_mat[:, i, j] = cosine_sim(x1_, x2_)
        return sim_mat

    def transform_with_cca(
        self,
        img_data: list[list[np.array]],
        lidar_data: list[list[np.array]],
        txt_data: list[list[np.array]],
    ) -> list[list[np.array]]:
        """Transform the data with cca or keep the data as is.

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
        data_lists: list[np.ndarray],
        idx_offset: int,
        num_workers: int = 8,
    ) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
        """Calculate the similarity matrices of all pairs of data, given in the args.

        Args:
            data_lists: list of data
            idx_offset: the index offset (calibration = train_size + test_size, test = train_size)
            num_workers: the number of workers to run in parallel

        Returns:
            sim_mat: the similarity matrices of a pair of data.
                key is the pair of indices in the original dataset,
                value is the similarity matrix and ground truth label.
        """
        (img_data, lidar_data, txt_data) = data_lists
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
        ds_retrieval_cls = KITTI_file_Retrieval()

        def process_chunk(
            chunk: np.ndarray,
        ) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
            process_sim_mat_cali = {}
            ds_indices_q = []
            ds_indices_r = []
            gt_labels = []
            x1_3x3_data = [[] for _ in range(3)]
            x2_3x3_data = [[] for _ in range(3)]
            i_lists = []
            j_lists = []

            for i in tqdm(
                chunk, desc=f"Processing chunk {chunk[0]}-{chunk[-1]}", leave=True
            ):
                for j in range(i, ds_size):
                    ds_idx_q = self.shuffle2idx[i + idx_offset]
                    ds_idx_r = self.shuffle2idx[j + idx_offset]
                    gt_label = ds_retrieval_cls.eval_retrieval_ids(ds_idx_q, ds_idx_r)

                    ds_indices_q.append(ds_idx_q)
                    ds_indices_r.append(ds_idx_r)
                    gt_labels.append(gt_label)
                    i_lists.append(i)
                    j_lists.append(j)

            x1_3x3_data[0].append(img_data[i_lists])
            x1_3x3_data[0].append(cca_img2lidar[i_lists])
            x1_3x3_data[0].append(cca_img2txt[i_lists])
            x2_3x3_data[0].append(img_data[j_lists])
            x2_3x3_data[0].append(cca_img2lidar[j_lists])
            x2_3x3_data[0].append(cca_img2txt[j_lists])

            x1_3x3_data[1].append(cca_lidar2img[i_lists])
            x1_3x3_data[1].append(lidar_data[i_lists])
            x1_3x3_data[1].append(cca_lidar2txt[i_lists])
            x2_3x3_data[1].append(cca_lidar2img[j_lists])
            x2_3x3_data[1].append(lidar_data[j_lists])
            x2_3x3_data[1].append(cca_lidar2txt[j_lists])

            x1_3x3_data[2].append(cca_txt2img[i_lists])
            x1_3x3_data[2].append(cca_txt2lidar[i_lists])
            x1_3x3_data[2].append(txt_data[i_lists])
            x2_3x3_data[2].append(cca_txt2img[j_lists])
            x2_3x3_data[2].append(cca_txt2lidar[j_lists])
            x2_3x3_data[2].append(txt_data[j_lists])

            print("Calculating similarity matrix...")
            sim_mat = self.calculate_similarity_matrix(x1_3x3_data, x2_3x3_data)
            for result_idx in range(sim_mat.shape[0]):
                process_sim_mat_cali[
                    (ds_indices_q[result_idx], ds_indices_r[result_idx])
                ] = (sim_mat[result_idx, :, :], gt_labels[result_idx])
            return process_sim_mat_cali

        sim_mat_cali = {}
        chunks = np.array_split(range(ds_size), num_workers)
        for chunk in chunks:
            process_sim_mat_cali = process_chunk(chunk)
            for k, v in process_sim_mat_cali.items():
                sim_mat_cali[k] = v
        return sim_mat_cali

    def get_cali_data(self) -> None:
        """Get the calibration data.

        Calculate and save the similarity matrix in the format of (sim_score, gt_label).
        Then, we run the calibration to get the conformal scores and obtain the prediction bands.
        """
        sim_mat_path = Path(
            self.cfg_dataset.paths.save_path,
            f"sim_mat_cali_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}.pkl",
        )
        if not sim_mat_path.exists():
            print("Generating calibration data...")
            img_data = self.imgdata["cali"]
            lidar_data = self.lidardata["cali"]
            txt_data = self.txtdata["cali"]
            idx_offset = self.train_size + self.test_size
            self.sim_mat_cali = self.calculate_pairs_data_similarity(
                (img_data, lidar_data, txt_data), idx_offset
            )
            # save the calibration data in the format of (sim_score, gt_label)
            with sim_mat_path.open("wb") as f:
                pickle.dump(self.sim_mat_cali, f)
        else:
            print("Loading calibration data...")
            self.sim_mat_cali = joblib.load(sim_mat_path.open("rb"))

        # set up prediction bands
        self.set_pred_band((3, 3))

    def get_test_data(self) -> None:
        """Get the test data. Create the similarity matrix in the format of (sim_score, gt_label).

        This step is extremely time-consuming, so we cache the similarity matrix in the pickle format
        and use batch processing to speed up the process.
        """
        super().get_test_data(
            (self.imgdata["test"], self.lidardata["test"], self.txtdata["test"])
        )

    def retrieve_one_data(
        self,
        con_mat: dict[tuple[int, int], tuple[np.ndarray, int]],
        idx_q: int,
        idx_offset: int,
        range_r: int,
        single_modal: bool = False,
    ) -> np.ndarray:
        """Retrieve one data from the similarity matrix.

        Args:
            con_mat: the conformal probability matrix.
            idx_q: the index of the query data
            idx_offset: the index offset (calibration = train_size + test_size, test = train_size)
            range_r: the range of the indices to retrieve. (test: (0, test_size), cali: (0, cali_size))
            single_modal: whether to retrieve the single modality data.

        Returns:
            retrieved_pairs: the retrieved pairs in the format of (modal_idx_1, modal_idx_2, conformal_prob, gt_label)
                and in descending order of the conformal probability.
        """
        retrieved_pairs = []
        ds_idx_q = self.shuffle2idx[idx_q + idx_offset]
        for idx_r in range(range_r):
            if idx_r == idx_q:  # cannot retrieve itself
                continue
            ds_idx_r = self.shuffle2idx[idx_r + idx_offset]
            # check if pair (ds_idx_q, ds_idx_r) is in the keys of con_mat
            if (ds_idx_q, ds_idx_r) in con_mat:
                idx_1, idx_2 = ds_idx_q, ds_idx_r
            else:
                idx_1, idx_2 = ds_idx_r, ds_idx_q
            assert (
                idx_1,
                idx_2,
            ) in con_mat, (
                f"({idx_1}, {idx_2}, {ds_idx_q}, {ds_idx_r}) is not in the con_mat"
            )
            retrieved_pairs.append(
                self.parse_retrieved_pairs(idx_1, idx_2, con_mat, single_modal, 9)
            )
        return retrieved_pairs
