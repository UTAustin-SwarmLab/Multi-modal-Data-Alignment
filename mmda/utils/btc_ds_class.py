"""Dataset class for any2any - BTC retrieval task."""

# ruff: noqa: S301, ERA001
import copy
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from mmda.utils.any2any_ds_class import BaseAny2AnyDataset
from mmda.utils.calibrate import (
    calibrate,
    con_mat_calibrate,
    get_calibration_scores_1st_stage,
    get_calibration_scores_2nd_stage,
)
from mmda.utils.cca_class import NormalizedCCA
from mmda.utils.sim_utils import batch_weighted_corr_sim


class BTCDataset(BaseAny2AnyDataset):
    """BTC dataset class for any2any retrieval task."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset. Size of the data is 52700 in total.

        In this task, we only consider text-to-image/audio retrieval.

        Args:
            cfg: configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.cfg_dataset = cfg["BTC"]
        self.shape = (2, 2)  # shape of the similarity matrix
        self.cali_size = 71
        self.train_size = 524  # no training data is needed for BTC
        self.test_size = 148
        self.save_tag = ""
        self.mapping_fn = np.max

    def load_data(self) -> None:
        """Load the data for retrieval."""
        # load data
        self.test_trend = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "blurred_test_timeseries.npy")
        ).reshape(-1, self.cfg_dataset.horizon)
        self.test_trend *= np.random.rand(*self.test_trend.shape)
        self.test_trend = np.float32(
            np.where(np.diff(self.test_trend, n=1, axis=1) > 0, 1, -1)
        )

        self.train_trend = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "blurred_train_timeseries.npy")
        ).reshape(-1, self.cfg_dataset.horizon)
        self.train_trend *= np.random.rand(*self.train_trend.shape)
        self.train_trend = np.float32(
            np.where(np.diff(self.train_trend, n=1, axis=1) > 0, 1, -1)
        )

        self.val_trend = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "blurred_val_timeseries.npy")
        ).reshape(-1, self.cfg_dataset.horizon)
        self.val_trend *= np.random.rand(*self.val_trend.shape)
        self.val_trend = np.float32(
            np.where(np.diff(self.val_trend, n=1, axis=1) > 0, 1, -1)
        )

        self.test_ts = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "test_timeseries.npy")
        ).reshape(-1, self.cfg_dataset.horizon)
        self.train_ts = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "train_timeseries.npy")
        ).reshape(-1, self.cfg_dataset.horizon)
        self.val_ts = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "val_timeseries.npy")
        ).reshape(-1, self.cfg_dataset.horizon)

        self.test_feat = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "test_tsfresh_features.npy")
        )
        self.train_feat = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "train_tsfresh_features.npy")
        )
        self.val_feat = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "val_tsfresh_features.npy")
        )

        self.test_cond = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "test_continuous_conditions.npy")
        )[:, ::24, :].reshape(self.test_size, -1)
        self.train_cond = np.load(
            Path(
                self.cfg_dataset.paths.dataset_path + "train_continuous_conditions.npy"
            )
        )[:, ::24, :].reshape(self.train_size, -1)
        self.val_cond = np.load(
            Path(self.cfg_dataset.paths.dataset_path + "val_continuous_conditions.npy")
        )[:, ::24, :].reshape(self.cali_size, -1)
        self.num_data = self.test_size + self.cali_size + self.train_size

        # print the details of the data
        # print(self.test_cond.shape, self.train_cond.shape, self.val_cond.shape)
        # print(self.test_trend.shape, self.train_trend.shape, self.val_trend.shape)
        # print(self.test_ts.shape, self.train_ts.shape, self.val_ts.shape)
        # print(self.test_feat.shape, self.train_feat.shape, self.val_feat.shape)

        # print(
        #     np.load(
        #         Path(
        #             self.cfg_dataset.paths.dataset_path + "tsfresh_feature_names.npy",
        #         ),
        #         allow_pickle=True,
        #     )
        # )

    def preprocess_retrieval_data(self) -> None:
        """Preprocess the data for retrieval."""
        super().preprocess_retrieval_data()
        # load data
        self.load_data()

        # train/test/calibration split only on the query size (59_800)
        self.cond = {
            "test": self.test_cond,
            "train": self.train_cond,
            "cali": self.val_cond,
        }
        self.trend = {
            "test": self.test_trend,
            "train": self.train_trend,
            "cali": self.val_trend,
        }
        self.ts = {
            "test": self.test_ts,
            "train": self.train_ts,
            "cali": self.val_ts,
        }
        self.feat = {
            "test": self.test_feat,
            "train": self.train_feat,
            "cali": self.val_feat,
        }

        # masking missing data in the test set. Mask the whole modality of an instance at a time.
        self.mask = {}
        if self.cfg_dataset.mask_ratio != 0:
            mask_num = int(self.test_size / self.cfg_dataset.mask_ratio * 2)
            # mask the text modality only since the audio modality already has missing data
            mask12 = np.random.choice(self.test_size, mask_num, replace=False)
            mask34 = np.random.choice(self.test_size, mask_num, replace=False)
            self.mask[0] = mask12[: mask_num // 2]
            self.mask[1] = mask12[mask_num // 2 :]
            self.mask[2] = mask34[: mask_num // 2]
            self.mask[3] = mask34[mask_num // 2 :]
        else:
            self.mask[0] = []
            self.mask[1] = []
            self.mask[2] = []
            self.mask[3] = []

    def check_correct_retrieval(self, q_idx: int, r_idx: int) -> bool:
        """Check if the retrieval is correct.

        Args:
            q_idx: the query index
            r_idx: the retrieved index

        Returns:
            True if the retrieval is correct, False otherwise
        """
        return q_idx == r_idx

    def train_crossmodal_similarity(self) -> None:
        """Train the cross-modal similarity, aka the CSA method."""
        cfg_dataset = self.cfg_dataset
        cca_save_path = Path(
            cfg_dataset.paths.save_path + "any2any_cca_" + "cond_ts.pkl"
        )
        self.cond_ts_cca = NormalizedCCA(
            min(list(self.train_cond.shape) + list(self.train_ts.shape))
        )
        if not cca_save_path.exists():
            (
                self.cca_cond_ts,
                self.cca_ts_cond,
                self.cond_ts_corr,
            ) = self.cond_ts_cca.fit_transform_train_data(
                self.cfg_dataset, self.cond["train"], self.ts["train"]
            )
            self.cond_ts_cca.save_model(cca_save_path)
        else:
            self.cond_ts_cca.load_model(cca_save_path)
            self.cca_cond_ts = self.cond_ts_cca.traindata1
            self.cca_ts_cond = self.cond_ts_cca.traindata2
            self.cond_ts_corr = self.cond_ts_cca.corr_coeff

        cca_save_path = Path(
            cfg_dataset.paths.save_path + "any2any_cca_" + "cond_feat.pkl"
        )
        self.cond_feat_cca = NormalizedCCA(
            min(list(self.train_cond.shape) + list(self.train_feat.shape))
        )
        if not cca_save_path.exists():
            self.cca_cond_feat, self.cca_feat_cond, self.cond_feat_corr = (
                self.cond_feat_cca.fit_transform_train_data(
                    self.cfg_dataset, self.cond["train"], self.feat["train"]
                )
            )
            self.cond_feat_cca.save_model(cca_save_path)
        else:
            self.cond_feat_cca.load_model(cca_save_path)
            self.cca_cond_feat = self.cond_feat_cca.traindata1
            self.cca_feat_cond = self.cond_feat_cca.traindata2
            self.cond_feat_corr = self.cond_feat_cca.corr_coeff

        cca_save_path = Path(
            cfg_dataset.paths.save_path + "any2any_cca_" + "trend_ts.pkl"
        )
        self.trend_ts_cca = NormalizedCCA(
            min(list(self.train_trend.shape) + list(self.train_ts.shape))
        )
        if not cca_save_path.exists():
            self.cca_trend_ts, self.cca_ts_trend, self.trend_ts_corr = (
                self.trend_ts_cca.fit_transform_train_data(
                    self.cfg_dataset, self.trend["train"], self.ts["train"]
                )
            )
            self.trend_ts_cca.save_model(cca_save_path)
        else:
            self.trend_ts_cca.load_model(cca_save_path)
            self.cca_trend_ts = self.trend_ts_cca.traindata1
            self.cca_ts_trend = self.trend_ts_cca.traindata2
            self.trend_ts_corr = self.trend_ts_cca.corr_coeff

        cca_save_path = Path(
            cfg_dataset.paths.save_path + "any2any_cca_" + "trend_feat.pkl"
        )
        self.trend_feat_cca = NormalizedCCA(
            min(self.train_trend.shape + self.train_feat.shape)
        )
        if not cca_save_path.exists():
            self.cca_trend_feat, self.cca_feat_trend, self.trend_feat_corr = (
                self.trend_feat_cca.fit_transform_train_data(
                    self.cfg_dataset, self.trend["train"], self.feat["train"]
                )
            )
            self.trend_feat_cca.save_model(cca_save_path)
        else:
            self.trend_feat_cca.load_model(cca_save_path)
            self.cca_trend_feat = self.trend_feat_cca.traindata1
            self.cca_feat_trend = self.trend_feat_cca.traindata2
            self.trend_feat_corr = self.trend_feat_cca.corr_coeff

    def calculate_similarity_matrix(
        self,
        x1: list[list[np.array]],
        x2: list[list[np.array]],
    ) -> np.ndarray:
        """Calculate the similarity matrix.

        Args:
            x1: the first data (not masked) shape is [2, 2], np: num_data, emb_dim)
            x2: the second data (not masked) shape is [2, 2], np: num_data, emb_dim)

        Returns:
            similarity_matrix: the similarity matrix of a pair of data shape is (num_data, 2, 2)
        """
        num_data = x1[0][0].shape[0]
        sim_mat = np.zeros((num_data, 2, 2))
        for i in range(2):
            for j in range(2):
                x1_ = x1[i][j]
                x2_ = x2[i][j]
                if np.any(np.isnan(x1_)) or np.any(np.isnan(x2_)):
                    sim_mat[:, i, j] = -1
                    msg = "NaN in the data, did you mask the data?"
                    raise ValueError(msg)
                if i == 0 and j == 0:
                    corr = self.cond_ts_corr
                    retrieval_dim = 50
                elif i == 0 and j == 1:
                    corr = self.cond_feat_corr
                    retrieval_dim = 50
                elif i == 1 and j == 0:
                    corr = self.trend_ts_corr
                    retrieval_dim = 100
                elif i == 1 and j == 1:
                    corr = self.trend_feat_corr
                    retrieval_dim = 100

                sim_mat[:, i, j] = batch_weighted_corr_sim(
                    x=x1_,
                    y=x2_,
                    corr=corr,
                    dim=retrieval_dim,
                )
        return sim_mat

    def transform_with_cca(
        self,
        cond_data: list[list[np.array]],
        trend_data: list[list[np.array]],
        ts_data: list[list[np.array]],
        feat_data: list[list[np.array]],
    ) -> list[list[np.array]]:
        """Transform the data with cca or keep the data as is.

        Args:
            cond_data: the condition data
            trend_data: the trend data
            ts_data: the time series data
            feat_data: the feature data

        Returns:
            cca_cond_ts: the cca transformed condition data to condition-time series space
            cca_cond_feat: the cca transformed condition data to condition-feature space
            cca_trend_ts: the cca transformed trend data to trend-time series space
            cca_trend_feat: the cca transformed trend data to trend-feature space
            cca_ts_cond: the cca transformed time series data to condition-time series space
            cca_feat_cond: the cca transformed feature data to condition-feature space
            cca_ts_trend: the cca transformed time series data to trend-time series space
            cca_feat_trend: the cca transformed feature data to trend-feature space
        """
        # cca transformation
        cca_cond_ts, cca_ts_cond = self.cond_ts_cca.transform_data(cond_data, ts_data)
        cca_cond_feat, cca_feat_cond = self.cond_feat_cca.transform_data(
            cond_data, feat_data
        )
        cca_trend_ts, cca_ts_trend = self.trend_ts_cca.transform_data(
            trend_data, ts_data
        )
        cca_trend_feat, cca_feat_trend = self.trend_feat_cca.transform_data(
            trend_data, feat_data
        )
        return (
            cca_cond_ts,
            cca_cond_feat,
            cca_trend_ts,
            cca_trend_feat,
            cca_ts_cond,
            cca_feat_cond,
            cca_ts_trend,
            cca_feat_trend,
        )

    def calculate_pairs_data_similarity(
        self,
        data_lists: list[np.ndarray],
        idx_offset: int = 0,
        num_workers: int = 1,
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
        assert idx_offset >= 0
        assert num_workers == 1  # no parallel processing for BTC
        (cond_data, trend_data, ts_data, feat_data) = data_lists
        (
            cca_cond_ts,
            cca_cond_feat,
            cca_trend_ts,
            cca_trend_feat,
            cca_ts_cond,
            cca_feat_cond,
            cca_ts_trend,
            cca_feat_trend,
        ) = self.transform_with_cca(cond_data, trend_data, ts_data, feat_data)
        ds_size = cond_data.shape[0]

        # calculate the similarity matrix, we do not mask the data here
        sim_mat_cali = {}
        ds_indices_q = []
        ds_indices_r = []
        gt_labels = []
        x1_2x2_data = [[] for _ in range(2)]
        x2_2x2_data = [[] for _ in range(2)]
        i_lists = []
        j_lists = []

        for i in tqdm(range(ds_size), desc="Processing data", leave=True):
            for j in range(ds_size):
                gt_label = self.check_correct_retrieval(i, j)

                ds_indices_q.append(i)
                ds_indices_r.append(j)
                gt_labels.append(gt_label)
                i_lists.append(i)
                j_lists.append(j)

        x1_2x2_data[0].append(cca_cond_ts[i_lists])
        x1_2x2_data[0].append(cca_cond_feat[i_lists])
        x2_2x2_data[0].append(cca_ts_cond[j_lists])
        x2_2x2_data[0].append(cca_feat_cond[j_lists])

        x1_2x2_data[1].append(cca_trend_ts[i_lists])
        x1_2x2_data[1].append(cca_trend_feat[i_lists])
        x2_2x2_data[1].append(cca_ts_trend[j_lists])
        x2_2x2_data[1].append(cca_feat_trend[j_lists])

        print("Calculating similarity matrix...")
        sim_mat = self.calculate_similarity_matrix(x1_2x2_data, x2_2x2_data)
        for result_idx in range(sim_mat.shape[0]):
            sim_mat_cali[(ds_indices_q[result_idx], ds_indices_r[result_idx])] = (
                sim_mat[result_idx, :, :],
                gt_labels[result_idx],
            )
        return sim_mat_cali

    def cal_test_conformal_prob(self) -> None:
        """Calculate the conformal probability for the test data.

        Args:
            shape: the shape of the similarity matrix
        """
        con_mat_test_path = Path(
            self.cfg_dataset.paths.save_path,
            f"con_mat_test_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}{self.save_tag}.json",
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
                        probs[i, j] = calibrate(sim_mat[i, j], self.scores_1st[(i, j)])
                self.con_mat_test[(idx_q, idx_r)] = (probs, gt_label)
            with con_mat_test_path.open("w") as f:
                json.dump(
                    {
                        f"{k[0]},{k[1]}": [v[0].tolist(), v[1]]
                        for k, v in self.con_mat_test.items()
                    },
                    f,
                )
        else:
            print("Loading conformal probabilities...")
            # load with pickle since it is faster than joblib (but less safe)
            with con_mat_test_path.open("r") as f:
                self.con_mat_test = json.load(f)
            # Convert keys back to tuples and values back to numpy arrays
            self.con_mat_test = {
                tuple(map(int, k.split(","))): (np.array(v[0]), v[1])
                for k, v in self.con_mat_test.items()
            }

        con_mat_test_miss_path = Path(
            self.cfg_dataset.paths.save_path,
            f"con_mat_test_miss_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}{self.save_tag}.json",
        )
        if not con_mat_test_miss_path.exists():
            self.con_mat_test_miss = copy.deepcopy(self.con_mat_test)
            for (idx_q, idx_r), (_, _) in tqdm(
                self.con_mat_test.items(),
                desc="Calculating conformal probabilities for missing data",
                leave=True,
            ):
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        if idx_q in self.mask[i] or idx_r in self.mask[j + 2]:
                            self.con_mat_test_miss[(idx_q, idx_r)][0][i, j] = -1
            with con_mat_test_miss_path.open("w") as f:
                json.dump(
                    {
                        f"{k[0]},{k[1]}": [v[0].tolist(), v[1]]
                        for k, v in self.con_mat_test_miss.items()
                    },
                    f,
                )
        else:
            print("Loading conformal probabilities for missing data...")
            # load with pickle since it is faster than joblib (but less safe)
            with con_mat_test_miss_path.open("r") as f:
                self.con_mat_test_miss = json.load(f)
            # Convert keys back to tuples and values back to numpy arrays
            self.con_mat_test_miss = {
                tuple(map(int, k.split(","))): (np.array(v[0]), v[1])
                for k, v in self.con_mat_test_miss.items()
            }

    def get_cali_data(self) -> None:
        """Get the calibration data.

        Calculate and save the similarity matrix in the format of (sim_score, gt_label).
        Then, we run the calibration to get the conformal scores and obtain the prediction bands.
        """
        sim_mat_path = Path(
            self.cfg_dataset.paths.save_path,
            f"sim_mat_cali_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}{self.save_tag}.json",
        )
        if not sim_mat_path.exists():
            print("Generating calibration data...")
            cond_data = self.cond["cali"]
            trend_data = self.trend["cali"]
            ts_data = self.ts["cali"]
            feat_data = self.feat["cali"]
            self.sim_mat_cali = self.calculate_pairs_data_similarity(
                (cond_data, trend_data, ts_data, feat_data)
            )
            # save the calibration data in the format of (sim_score, gt_label)
            with sim_mat_path.open("w") as f:
                json.dump(
                    {
                        f"{k[0]},{k[1]}": [v[0].tolist(), v[1]]
                        for k, v in self.sim_mat_cali.items()
                    },
                    f,
                )
        else:
            print("Loading calibration data...")
            with sim_mat_path.open("r") as f:
                self.sim_mat_cali = json.load(f)
            # Convert keys back to tuples and values back to numpy arrays
            self.sim_mat_cali = {
                tuple(map(int, k.split(","))): (np.array(v[0]), v[1])
                for k, v in self.sim_mat_cali.items()
            }

        # set up prediction bands
        self.set_pred_band()

    def set_pred_band(self) -> None:
        """Set up the 1st stage prediction bands for the calibration."""
        self.scores_1st = {}
        print("1st stage prediction bands")
        # calculate the calibration scores and conformal scores for all pairs of modalities
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.scores_1st[(i, j)] = get_calibration_scores_1st_stage(
                    self.sim_mat_cali, i, j
                )[0]
        cali_con_mat = con_mat_calibrate(self.sim_mat_cali, self.scores_1st)
        self.scores_2nd = get_calibration_scores_2nd_stage(
            cali_con_mat, self.mapping_fn
        )[0]

        con_mat_cali_miss = deepcopy(cali_con_mat)
        # mask data in the missing calibration conformal matrix
        for (idx_q, idx_r), (_, _) in tqdm(
            cali_con_mat.items(),
            desc="Masking conformal probabilities",
            leave=True,
        ):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if idx_q in self.mask[i] or idx_r in self.mask[j + 2]:
                        con_mat_cali_miss[(idx_q, idx_r)][0][i, j] = -1
        self.scores_2nd_miss = get_calibration_scores_2nd_stage(
            con_mat_cali_miss, self.mapping_fn
        )[0]

    def get_test_data(self) -> None:
        """Get the test data. Create the similarity matrix in the format of (sim_score, gt_label).

        This step is extremely time-consuming, so we cache the similarity matrix in the pickle format
        and use batch processing to speed up the process.
        """
        super().get_test_data(
            (self.cond["test"], self.trend["test"], self.ts["test"], self.feat["test"])
        )

    def retrieve_one_data(
        self,
        con_mat: dict[tuple[int, int], tuple[np.ndarray, int]],
        idx_q: int,
        idx_offset: int,
        range_r: int,
        single_modal: bool = False,
        scores_2nd: list[float] | None = None,
    ) -> np.ndarray:
        """Retrieve one data from the similarity matrix.

        Args:
            con_mat: the conformal probability matrix.
            idx_q: the index of the query data
            idx_offset: the index offset (calibration = train_size + test_size, test = train_size)
            range_r: the range of the indices to retrieve. (test: (0, test_size), cali: (0, cali_size))
            single_modal: whether to retrieve the single modality data.
            scores_2nd: the second scores for the retrieval.

        Returns:
            retrieved_pairs: the retrieved pairs in the format of (modal_idx_1, modal_idx_2, conformal_prob, gt_label)
                and in descending order of the conformal probability.
        """
        assert idx_offset >= 0  # no offset for BTC
        retrieved_pairs = []
        for idx_r in range(range_r):
            # check if pair (idx_q, idx_r) is in the keys of con_mat
            retrieved_pairs.append(
                self.parse_retrieved_pairs(
                    idx_q,
                    idx_r,
                    con_mat,
                    single_modal,
                    scores_2nd,
                )
            )
        return retrieved_pairs
