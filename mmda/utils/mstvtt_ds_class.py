"""Dataset class for any2any - msrvtt retrieval task."""

import copy
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from mmda.utils.any2any_ds_class import BaseAny2AnyDataset
from mmda.utils.calibrate import (
    calibrate,
)
from mmda.utils.dataset_utils import load_msrvtt


def process_similarity_pair(inputs: tuple) -> dict:
    """Processes similarity pairs for retrieval tasks.

    Args:
        inputs: all the inputs.
        check_correct_retrieval (callable): A function to check the ground truth label for a given pair.
        txt2img_data (np.ndarray): The text-to-image data.
        img2txt_data (np.ndarray): The image-to-text data.
        txt2audio_data (np.ndarray): The text-to-audio data.
        audio2txt_data (np.ndarray): The audio-to-text data.
        range_idx_q (range): The index of the query.
        r_size (int): The size of the retrieval range.
        idx_offset (int): The offset for the indices.
        step_size (int): The step size for the retrieval indices.

    Returns:
        dict: A dictionary containing the cosine similarities and ground truth labels for each pair.
    """
    (
        check_correct_retrieval,
        txt2img_data,
        img2txt_data,
        txt2audio_data,
        audio2txt_data,
        range_idx_q,
        r_size,
        idx_offset,
        step_size,
    ) = inputs
    sim_mat_i = {}
    for idx_q in tqdm(range_idx_q):
        for j in range(r_size):
            gt_label = check_correct_retrieval(idx_q + idx_offset, j)
            cosine_sim_txt2img = np.sum(
                txt2img_data[idx_q - range_idx_q[0]] * img2txt_data[j]
            )
            cosine_sim_txt2audio = np.nansum(
                txt2audio_data[idx_q - range_idx_q[0]] * audio2txt_data[j]
            )
            if cosine_sim_txt2audio == 0:
                cosine_sim_txt2audio = -1
            sim_mat_i[(idx_q + idx_offset, j)] = (
                np.array([cosine_sim_txt2img, cosine_sim_txt2audio]).reshape(1, -1),
                gt_label,
            )
    return sim_mat_i


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
        self.text2img = "clip"
        self.text2audio = "clap"

        self.shape = (1, 2)  # shape of the similarity matrix
        self.cali_size = 3800
        self.train_size = 53_000  # no training data is needed for MSRVTT
        self.test_size = 3_000
        self.step_size = 20  # 20 duplicates of different captions of a video
        self.img2txt_encoder = self.cfg_dataset.img_encoder
        self.audio2txt_encoder = self.cfg_dataset.audio_encoder
        self.save_tag = f"{self.img2txt_encoder}_{self.audio2txt_encoder}"

    def load_data(self) -> None:
        """Load the data for retrieval."""
        self.sen_ids, self.captions, self.video_info_sen_order, self.video_dict = (
            load_msrvtt(self.cfg_dataset)
        )
        with Path(self.cfg_dataset.paths.save_path, "MSRVTT_ref_video_ids.pkl").open(
            "rb"
        ) as file:
            self.ref_id_order = pickle.load(file)  # noqa: S301
        null_audio_idx = [
            self.video_dict[video_id]["audio_np"] is None
            for video_id in self.ref_id_order
        ]

        # get video idx which has no audio. 355 in total.
        null_audio_idx = []
        for idx, video_info in enumerate(self.video_info_sen_order):
            if video_info["audio_np"] is None and idx % self.step_size == 0:
                null_audio_idx.append(int(idx / self.step_size))
        # load data
        with Path(
            self.cfg_dataset.paths.save_path
            + f"MSRVTT_text_emb_{self.img2txt_encoder}.pkl"
        ).open("rb") as file:
            self.txt2img_emb = pickle.load(file)  # (59800, 1280) # noqa: S301
        with Path(
            self.cfg_dataset.paths.save_path
            + f"MSRVTT_video_emb_{self.img2txt_encoder}.pkl"
        ).open("rb") as file:
            self.img2txt_emb = pickle.load(file)  # noqa: S301
        print(self.img2txt_emb.shape)
        with Path(
            self.cfg_dataset.paths.save_path
            + f"MSRVTT_text_emb_{self.audio2txt_encoder}.pkl"
        ).open("rb") as file:
            self.txt2audio_emb = pickle.load(file)  # (59800, 512) # noqa: S301
        with Path(
            self.cfg_dataset.paths.save_path
            + f"MSRVTT_audio_emb_{self.audio2txt_encoder}.pkl"
        ).open("rb") as file:
            self.audio2txt_emb = pickle.load(file)  # (???, 512) # noqa: S301
        print(self.audio2txt_emb.shape)

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
        # get the avg of video (2 frame) embeddings, as it does not affect the cosine similarity
        self.img2txt_emb = (
            self.img2txt_emb[:, : int(self.img2txt_emb.shape[1] / 2)]
            + self.img2txt_emb[:, int(self.img2txt_emb.shape[1] / 2) :]
        ) / 2
        self.txt2audio_emb = self.txt2audio_emb / np.linalg.norm(
            self.txt2audio_emb, axis=1, keepdims=True
        )
        self.audio2txt_emb = self.audio2txt_emb / np.linalg.norm(
            self.audio2txt_emb, axis=1, keepdims=True
        )
        # handle missing audio in videos
        self.audio2txt_emb[null_audio_idx] = np.nan

    def preprocess_retrieval_data(self) -> None:
        """Preprocess the data for retrieval."""
        # load data
        self.load_data()
        self.num_data = self.txt2img_emb.shape[0]
        assert (
            self.test_size + self.cali_size + self.train_size == self.num_data
        ), f"{self.test_size} + {self.cali_size} + {self.train_size} != {self.num_data}"

        # train/test/calibration split only on the query size (59_800)
        # Shuffle the array to ensure randomness
        idx = np.arange(self.num_data)  # 2990
        txt_test_idx = idx[self.train_size : self.cali_size]
        txt_cali_idx = idx[-self.cali_size :]
        self.txt2img_emb = {
            "test": self.txt2img_emb[txt_test_idx],
            "cali": self.txt2img_emb[txt_cali_idx],
        }
        self.img2txt_emb = {
            "test": self.img2txt_emb,
            "cali": self.img2txt_emb,
        }
        self.txt2audio_emb = {
            "test": self.txt2audio_emb[txt_test_idx],
            "cali": self.txt2audio_emb[txt_cali_idx],
        }
        self.audio2txt_emb = {
            "test": self.audio2txt_emb,
            "cali": self.audio2txt_emb,
        }
        # masking missing data in the test set. Mask the whole modality of an instance at a time.
        if self.cfg_dataset.mask_ratio != 0:
            mask_num = int(self.test_size / self.cfg_dataset.mask_ratio)
            self.mask = {}  # modality -> masked idx
            mask_idx = np.random.choice(self.test_size, mask_num * 2, replace=False)
            self.mask[0] = mask_idx[:mask_num]
            self.mask[1] = mask_idx[mask_num:]
        else:
            self.mask = {}
            self.mask[0] = []
            self.mask[1] = []

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
            == self.ref_id_order[r_idx]["video_id"]
        )

    def calculate_pairs_data_similarity(
        self,
        data_lists: list[np.ndarray],
        idx_offset: int,
        num_workers: int = 2,
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
            f"Calculating similarity matrix... num_workers={num_workers} is used here."
        )
        (txt2img_data, img2txt_data, txt2audio_data, audio2txt_data) = data_lists
        q_size = txt2img_data.shape[0]
        r_size = img2txt_data.shape[0]
        range_idx_qs = [
            range(i * q_size // num_workers, (i + 1) * q_size // num_workers)
            for i in range(num_workers)
        ]
        with Pool(num_workers) as p:
            sim_mat_chunks = p.map(
                process_similarity_pair,
                [
                    (
                        self.check_correct_retrieval,
                        txt2img_data[range_idx_q],
                        img2txt_data,
                        txt2audio_data[range_idx_q],
                        audio2txt_data,
                        range_idx_q,
                        r_size,
                        idx_offset,
                        self.step_size,
                    )
                    for range_idx_q in range_idx_qs
                ],
            )
        sim_mat = {}
        for sim_mat_dict in sim_mat_chunks:
            for k, v in sim_mat_dict.items():
                sim_mat[k] = v
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
            self.sim_mat_cali = pickle.load(sim_mat_path.open("rb"))  # noqa: S301

        # set up prediction bands
        self.set_pred_band()

    def cal_test_conformal_prob(self) -> None:
        """Calculate the conformal probability for the test data.

        Args:
            shape: the shape of the similarity matrix
        """
        con_mat_test_path = Path(
            self.cfg_dataset.paths.save_path,
            f"con_mat_test_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}{self.save_tag}.pkl",
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
                        probs[i, j] = calibrate(sim_mat[i, j], self.scores_1st[(i, j)])
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
            f"con_mat_test_miss_{self.cfg_dataset.retrieval_dim}_{self.cfg_dataset.mask_ratio}{self.save_tag}.pkl",
        )
        if not con_mat_test_miss_path.exists():
            self.con_mat_test_miss = copy.deepcopy(self.con_mat_test)
            if self.mask[0] != []:
                for (idx_q, idx_r), (_, _) in tqdm(
                    self.con_mat_test.items(),
                    desc="Calculating conformal probabilities for missing data",
                    leave=True,
                ):
                    for j in range(self.shape[1]):
                        if idx_r in self.mask[j]:
                            self.con_mat_test_miss[(idx_q, idx_r)][0][0, j] = -1
                with con_mat_test_miss_path.open("wb") as f:
                    pickle.dump(self.con_mat_test_miss, f)
        else:
            print("Loading conformal probabilities for missing data...")
            # load with pickle since it is faster than joblib (but less safe)
            with con_mat_test_miss_path.open("rb") as f:
                self.con_mat_test_miss = pickle.load(f)  # noqa: S301

    def get_test_data(self) -> None:
        """Get the test data. Create the similarity matrix in the format of (sim_score, gt_label).

        This step is extremely time-consuming, so we cache the similarity matrix in the pickle format
        and use batch processing to speed up the process.
        """
        super().get_test_data(
            (
                self.txt2img_emb["test"],
                self.img2txt_emb["test"],
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
        for ds_idx_r in range(self.img2txt_emb.shape[0]):
            retrieved_pairs.append(
                self.parse_retrieved_pairs(
                    ds_idx_q, ds_idx_r, con_mat, single_modal, scores_2nd
                )
            )
        return retrieved_pairs
