"""Dataset class for any2any retrieval task."""

import pickle
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm

from mmda.utils.calibrate import (
    calibrate,
    con_mat_calibrate,
    get_calibration_scores_1st_stage,
    get_calibration_scores_2nd_stage,
)
from mmda.utils.liploc_model import get_top_k


class BaseAny2AnyDataset:
    """Base class for any2any retrieval dataset."""

    def __init__(self) -> None:
        """Initialize the dataset."""
        self.mapping_fn = np.mean
        self.save_tag = ""

    def preprocess_retrieval_data(self) -> None:
        """Preprocess the data for retrieval."""

    def train_crossmodal_similarity(self) -> None:
        """Train the cross-modal similarity, aka the CSA method."""

    def get_cali_data(self) -> None:
        """Get the calibration data."""

    def set_pred_band(self) -> None:
        """Set up the 1st stage prediction bands for the calibration."""
        self.scores_1st = {}
        print("Calculating calibration scores...")
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
            desc="Masking conformal probabilities for missing data",
            leave=True,
        ):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if idx_q in self.mask[i] and idx_r in self.mask[j]:
                        con_mat_cali_miss[(idx_q, idx_r)][0][i, j] = -1
        self.scores_2nd_miss = get_calibration_scores_2nd_stage(
            con_mat_cali_miss, self.mapping_fn
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
            f"con_mat_test_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}{self.save_tag}.pkl",
        ).exists():
            print(
                "Since the conformal probabilities are already calculated, we skip the process of loading test data."
            )
            return

        sim_mat_test_path = Path(
            self.cfg_dataset.paths.save_path,
            f"sim_mat_test_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}{self.save_tag}.pkl",
        )
        if not sim_mat_test_path.exists():
            print("Generating test data...")
            idx_offset = self.train_size
            self.sim_mat_test = self.calculate_pairs_data_similarity(
                data_lists,
                idx_offset,
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

    def parse_retrieved_pairs(
        self,
        ds_idx_q: int,
        ds_idx_r: int,
        con_mat: dict,
        single_modal: bool = False,
        cali_scores: list[float] | None = None,
    ) -> tuple[int, int, float, int]:
        """Parse the retrieved pairs.

        Args:
            ds_idx_q: the index of the query data
            ds_idx_r: the index of the retrieved data
            con_mat: the conformal probability matrix
            single_modal: whether to retrieve the single modality data
            cali_scores: the calibration scores

        Returns:
            ds_idx_q: the index of the query data
            ds_idx_r: the index of the retrieved data
            conformal_prob: the conformal probability (sometimes the mean of the top-k conformal probabilities)
            gt_label: the ground truth label
        """
        probs = con_mat[(ds_idx_q, ds_idx_r)][0]
        # single modality retrieval
        if single_modal:
            return (ds_idx_q, ds_idx_r, probs, con_mat[(ds_idx_q, ds_idx_r)][1])
        # full modality retrieval
        probs_filtered = probs[probs != -1]  # Ignore -1 entries
        if len(probs_filtered) > 0:
            prob_filtered = self.mapping_fn(probs_filtered)
            cali_prob = calibrate(prob_filtered, cali_scores)
        else:
            cali_prob = 0  # Default value if all entries are -1
        return (
            ds_idx_q,
            ds_idx_r,
            cali_prob,  # conformal_prob
            con_mat[(ds_idx_q, ds_idx_r)][1],  # gt_label
        )

    def retrieve_data(  # noqa: PLR0915
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
        scores_2nd = self.scores_2nd_miss if mode == "miss" else self.scores_2nd
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
                    con_mat,
                    idx_q,
                    self.train_size,
                    self.test_size,
                    single_modal=False,
                    scores_2nd=scores_2nd,
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
        recalls = {
            (i, j): [] for i in range(self.shape[0]) for j in range(self.shape[1])
        }
        precisions = {
            (i, j): [] for i in range(self.shape[0]) for j in range(self.shape[1])
        }
        maps = {(i, j): [] for i in range(self.shape[0]) for j in range(self.shape[1])}
        from concurrent.futures import ProcessPoolExecutor

        def process_retrieval(inputs: tuple[callable, int]) -> list:
            retrieve_fn, idx_q = inputs
            retrieved_pairs = retrieve_fn(
                con_mat, idx_q, self.train_size, self.test_size, single_modal=True
            )
            results = []
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    modal_pair = (i, j)
                    retrieved_pairs_ij = sorted(
                        retrieved_pairs, key=lambda x: x[2][modal_pair], reverse=True
                    )
                    top_k_hit = get_top_k(retrieved_pairs_ij, k=5)
                    recall_k = 1 if any(top_k_hit) else 0
                    results.append((modal_pair, recall_k))
            return results

        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(process_retrieval, (self.retrieve_one_data, idx_q))
                for idx_q in range(self.test_size)
            ]
            for future in tqdm(futures, desc=f"Retrieving {mode} data", leave=True):
                results = future.result()
                for modal_pair, recall_k in results:
                    recalls[modal_pair].append(recall_k)

        for modal_pair in recalls:
            recalls[modal_pair] = np.mean(recalls[modal_pair])
            precisions[modal_pair] = 0  # np.mean(precisions[modal_pair])
            maps[modal_pair] = 0  # np.mean(maps[modal_pair])
        return maps, precisions, recalls
