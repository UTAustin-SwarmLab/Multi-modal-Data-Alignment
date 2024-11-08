"""Dataset class for any2any - msrvtt retrieval task."""

# ruff: noqa: S301
import copy
import json
import pickle
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from mmda.utils.any2any_ds_class import BaseAny2AnyDataset
from mmda.utils.calibrate import (
    calibrate,
)
from mmda.utils.dataset_utils import load_msrvtt


class MSRVTTDataset(BaseAny2AnyDataset):
    """MSRVTT dataset class for any2any retrieval task."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset. Size of the data is 52700 in total.

        In this task, we only consider text-to-image/audio retrieval.

        Args:
            cfg: configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.cfg_dataset = cfg["MSRVTT"]
        self.shape = (1, 2)  # shape of the similarity matrix
        self.cali_size = 460
        self.train_size = 6_500  # no training data is needed for MSRVTT
        self.test_size = 5_000
        self.query_step = 5  # 59800 / 5 = 11960
        # 24403 / 5 = 4880 and 24403 / 2 = 12201
        self.ref_step = 5 if self.cfg_dataset.audio_encoder == "clap" else 2
        self.img2txt_encoder = self.cfg_dataset.img_encoder
        self.audio2txt_encoder = self.cfg_dataset.audio_encoder
        self.save_tag = f"_{self.img2txt_encoder}_{self.audio2txt_encoder}"

    def load_data(self) -> None:
        """Load the data for retrieval."""
        _, _, self.video_info_sen_order, _ = load_msrvtt(self.cfg_dataset)
        with Path(self.cfg_dataset.paths.save_path, "MSRVTT_id_order.pkl").open(
            "rb"
        ) as f:
            self.ref_id_order = pickle.load(f)[:: self.ref_step]
        self.video_info_sen_order = self.video_info_sen_order[:: self.query_step]
        with Path(self.cfg_dataset.paths.save_path, "MSRVTT_null_audio.pkl").open(
            "rb"
        ) as f:
            # get video idx which has no audio. 355 in total. list of bool in ref_id_order
            self.null_audio_idx = pickle.load(f)[:: self.ref_step]

        # load data
        with Path(
            self.cfg_dataset.paths.save_path
            + f"MSRVTT_text_emb_{self.img2txt_encoder}.pkl"
        ).open("rb") as file:
            self.txt2img_emb = pickle.load(file)[:: self.query_step]
        with Path(
            self.cfg_dataset.paths.save_path
            + f"MSRVTT_video_emb_{self.img2txt_encoder}.pkl"
        ).open("rb") as file:
            self.img2txt_emb = pickle.load(file)[:: self.ref_step]
        with Path(
            self.cfg_dataset.paths.save_path
            + f"MSRVTT_text_emb_{self.audio2txt_encoder}.pkl"
        ).open("rb") as file:
            self.txt2audio_emb = pickle.load(file)[:: self.query_step]
        with Path(
            self.cfg_dataset.paths.save_path
            + f"MSRVTT_audio_emb_{self.audio2txt_encoder}.pkl"
        ).open("rb") as file:
            if self.audio2txt_encoder == "clap":
                self.audio2txt_emb = pickle.load(file)
            else:
                self.audio2txt_emb = pickle.load(file)[:: self.ref_step]
        self.img2txt_emb = self.img2txt_emb[: self.audio2txt_emb.shape[0]]
        self.ref_id_order = self.ref_id_order[: self.audio2txt_emb.shape[0]]
        self.null_audio_idx = self.null_audio_idx[: self.audio2txt_emb.shape[0]]
        assert (
            self.audio2txt_emb.shape[0] == self.img2txt_emb.shape[0]
        ), f"{self.audio2txt_emb.shape}, {self.img2txt_emb.shape}"
        assert (
            self.txt2audio_emb.shape[0] == self.txt2img_emb.shape[0]
        ), f"{self.txt2audio_emb.shape}, {self.txt2img_emb.shape}"

        # normalize all the embeddings to have unit norm using L2 normalization
        self.txt2img_emb = self.txt2img_emb / np.linalg.norm(
            self.txt2img_emb, axis=1, keepdims=True
        )
        self.img2txt_emb = self.img2txt_emb / np.linalg.norm(
            self.img2txt_emb, axis=1, keepdims=True
        )
        self.txt2audio_emb = self.txt2audio_emb / np.linalg.norm(
            self.txt2audio_emb, axis=1, keepdims=True
        )
        self.audio2txt_emb = self.audio2txt_emb / np.linalg.norm(
            self.audio2txt_emb, axis=1, keepdims=True
        )
        self.num_data = self.txt2img_emb.shape[0]

        # handle missing audio in videos
        self.audio2txt_emb[self.null_audio_idx] = np.nan  # 2848 missing out of 24403
        print(f"Number of videos with no audio: {np.sum(self.null_audio_idx)}")

        # check the length of the reference order
        assert (
            len(self.ref_id_order) == self.audio2txt_emb.shape[0]
        ), f"{len(self.ref_id_order)} != {self.audio2txt_emb.shape[0]}"
        assert (
            len(self.video_info_sen_order) == self.num_data
        ), f"{len(self.video_info_sen_order)} != {self.num_data}"

    def preprocess_retrieval_data(self) -> None:
        """Preprocess the data for retrieval."""
        super().preprocess_retrieval_data()
        # load data
        self.load_data()
        assert (
            self.test_size + self.cali_size + self.train_size == self.num_data
        ), f"{self.test_size} + {self.cali_size} + {self.train_size} != {self.num_data}"

        # train/test/calibration split only on the query size (59_800)
        idx = np.arange(self.num_data)
        txt_test_idx = idx[self.train_size : -self.cali_size]
        txt_cali_idx = idx[-self.cali_size :]
        self.txt2img_emb = {
            "test": self.txt2img_emb[txt_test_idx],
            "cali": self.txt2img_emb[txt_cali_idx],
        }
        self.txt2audio_emb = {
            "test": self.txt2audio_emb[txt_test_idx],
            "cali": self.txt2audio_emb[txt_cali_idx],
        }
        # masking missing data in the test set. Mask the whole modality of an instance at a time.
        self.mask = {}
        self.mask[1] = []
        if self.cfg_dataset.mask_ratio != 0:
            mask_num = int(self.test_size / self.cfg_dataset.mask_ratio)
            # mask the text modality only since the audio modality already has missing data
            self.mask[0] = np.random.choice(self.test_size, mask_num, replace=False)
        else:
            self.mask[0] = []

    def check_correct_retrieval(self, q_idx: int, r_idx: int) -> bool:
        """Check if the retrieval is correct.

        Args:
            q_idx: the query index
            r_idx: the retrieved index

        Returns:
            True if the retrieval is correct, False otherwise
        """
        return self.video_info_sen_order[q_idx]["video_id"] == self.ref_id_order[r_idx]

    def calculate_pairs_data_similarity(
        self,
        data_lists: list[np.ndarray],
        idx_offset: int,
    ) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
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
        (txt2img_data, img2txt_data, txt2audio_data, audio2txt_data) = data_lists
        q_size = txt2img_data.shape[0]
        r_size = img2txt_data.shape[0]
        sim_mat = {}
        for idx_q in tqdm(range(q_size), desc="Calculating similarity matrix"):
            for idx_r in range(r_size):
                gt_label = self.check_correct_retrieval(idx_q + idx_offset, idx_r)
                cosine_sim_txt2img = np.sum(txt2img_data[idx_q] * img2txt_data[idx_r])
                cosine_sim_txt2audio = np.nansum(
                    txt2audio_data[idx_q] * audio2txt_data[idx_r]
                )
                if cosine_sim_txt2audio == 0:
                    cosine_sim_txt2audio = -1
                sim_mat[(idx_q + idx_offset, idx_r)] = (
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
            f"sim_mat_cali_{self.cfg_dataset.mask_ratio}{self.save_tag}.json",
        )
        if not sim_mat_path.exists():
            print("Generating calibration data...")
            txt2img_data = self.txt2img_emb["cali"]
            img2txt_data = self.img2txt_emb
            txt2audio_data = self.txt2audio_emb["cali"]
            audio2txt_data = self.audio2txt_emb
            idx_offset = self.train_size + self.test_size
            self.sim_mat_cali = self.calculate_pairs_data_similarity(
                (txt2img_data, img2txt_data, txt2audio_data, audio2txt_data),
                idx_offset,
            )
            # save the calibration data in the format of (sim_score, gt_label)
            sim_mat_cali_json = {
                f"{k[0]},{k[1]}": [v[0].tolist(), v[1]]
                for k, v in self.sim_mat_cali.items()
            }
            with sim_mat_path.open("w") as f:
                json.dump(sim_mat_cali_json, f)
        else:
            print("Loading calibration data...")
            with sim_mat_path.open("r") as f:
                self.sim_mat_cali = json.load(f)
            self.sim_mat_cali = {
                tuple(map(int, k.split(","))): (np.array(v[0]), v[1])
                for k, v in self.sim_mat_cali.items()
            }

        # set up prediction bands
        self.set_pred_band()

    def cal_test_conformal_prob(self) -> None:  # noqa: PLR0912, C901
        """Calculate the conformal probability for the test data.

        Args:
            shape: the shape of the similarity matrix
        """
        con_mat_test_path = Path(
            self.cfg_dataset.paths.save_path,
            f"con_mat_test_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}{self.save_tag}.json",
        )
        if not con_mat_test_path.exists():
            self.con_mat_test = {}
            for (idx_q, idx_r), (sim_mat, gt_label) in tqdm(
                self.sim_mat_test.items(),
                desc="Calculating conformal probabilities",
                leave=True,
            ):
                probs = np.zeros(self.shape)
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        probs[i][j] = calibrate(sim_mat[i, j], self.scores_1st[(i, j)])
                self.con_mat_test[(idx_q, idx_r)] = (probs, int(gt_label))
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
            with con_mat_test_path.open("r") as f:
                con_mat_test = json.load(f)
            # Convert keys back to tuples and values back to numpy arrays
            self.con_mat_test = {
                tuple(map(int, k.split(","))): (np.array(v[0]), v[1])
                for k, v in con_mat_test.items()
            }
        con_mat_test_miss_path = Path(
            self.cfg_dataset.paths.save_path,
            f"con_mat_test_miss_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}{self.save_tag}.json",
        )
        if not con_mat_test_miss_path.exists():
            self.con_mat_test_miss = copy.deepcopy(self.con_mat_test)
            mask_modal = []
            if len(self.mask[0]) > 0:
                mask_modal.append(0)
            if len(self.mask[1]) > 0:
                mask_modal.append(1)
            if len(mask_modal) > 0:
                for (idx_q, idx_r), (_, _) in tqdm(
                    self.con_mat_test.items(),
                    desc="Calculating conformal probabilities for missing data",
                    leave=True,
                ):
                    for j in mask_modal:
                        if idx_r in self.mask[j]:
                            self.con_mat_test_miss[(idx_q, idx_r)][0][0, j] = -1
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
            with con_mat_test_miss_path.open("r") as f:
                self.con_mat_test_miss = json.load(f)
            # Convert keys back to tuples and values back to numpy arrays
            self.con_mat_test_miss = {
                tuple(map(int, k.split(","))): (np.array(v[0]), v[1])
                for k, v in self.con_mat_test_miss.items()
            }

    def get_test_data(self) -> None:
        """Get the test data. Create the similarity matrix in the format of (sim_score, gt_label).

        This step is extremely time-consuming, so we cache the similarity matrix in the json format
        and use batch processing to speed up the process.
        """
        super().get_test_data(
            (
                self.txt2img_emb["test"],
                self.img2txt_emb,
                self.txt2audio_emb["test"],
                self.audio2txt_emb,
            )
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
            scores_2nd: the calibration scores

        Returns:
            retrieved_pairs: the retrieved pairs in the format of (idx_q, idx_r, conformal_prob, gt_label)
                and in descending order of the conformal probability.
        """
        range_r = range_r + 1  # unused
        retrieved_pairs = []
        ds_idx_q = idx_q + idx_offset
        for ds_idx_r in range(len(self.ref_id_order)):
            retrieved_pairs.append(
                self.parse_retrieved_pairs(
                    ds_idx_q, ds_idx_r, con_mat, single_modal, scores_2nd
                )
            )
        return retrieved_pairs
