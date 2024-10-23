"""Dataset class for any2any - KITTI retrieval task."""

import copy
import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from mmda.utils.any2any_ds_class import BaseAny2AnyDataset
from mmda.utils.calibrate import calibrate
from mmda.utils.cca_class import NormalizedCCA
from mmda.utils.data_utils import load_three_encoder_data
from mmda.utils.liploc_model import Args, KITTI_file_Retrieval
from mmda.utils.sim_utils import batch_weighted_corr_sim, cosine_sim


class KITTIDataset(BaseAny2AnyDataset):
    """KITTI dataset class for any2any retrieval task."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        super().__init__()
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
        self.save_tag = f"_thres_{Args.threshold_dist}_shuffle_{self.shuffle_step}"

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
            idx = np.arange(self.num_data)
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
                        if idx_q in self.mask[i] or idx_r in self.mask[j]:
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
            img_data = self.imgdata["cali"]
            lidar_data = self.lidardata["cali"]
            txt_data = self.txtdata["cali"]
            idx_offset = self.train_size + self.test_size
            self.sim_mat_cali = self.calculate_pairs_data_similarity(
                (img_data, lidar_data, txt_data), idx_offset
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
            retrieved_pairs.append(
                self.parse_retrieved_pairs(
                    idx_1,
                    idx_2,
                    con_mat,
                    single_modal,
                    scores_2nd,
                )
            )
        return retrieved_pairs
