"""Dataset class for any2any retrieval task."""

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from mmda.utils.data_utils import load_three_encoder_data
from mmda.utils.liploc_model import Args, KITTI_file_Retrieval, get_top_k
from mmda.utils.sim_utils import cosine_sim


class KITTIEMMADataset:
    """KITTI dataset class for EMMA multimodal retrieval task."""

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def train_crossmodal_similarity(  # noqa: C901, PLR0912
        self, max_epoch: int
    ) -> None:
        """Train the cross-modal similarity, aka the CSA method."""
        data_loader = self.get_joint_dataloader(batch_size=256, num_workers=4)
        self.define_fc_networks(output_dim=256)
        self.img_fc.to(self.device)
        self.lidar_fc.to(self.device)
        self.txt_fc.to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.img_fc.parameters())
            + list(self.lidar_fc.parameters())
            + list(self.txt_fc.parameters()),
            lr=0.001,
        )

        model_path = Path(self.cfg["KITTI"].paths.save_path) / "models"
        model_path.mkdir(parents=True, exist_ok=True)
        ds_retrieval_cls = KITTI_file_Retrieval()

        for epoch in range(max_epoch):
            for _, (img, lidar, txt, orig_idx) in enumerate(data_loader):
                bs = img.shape[0]
                img_embed = self.img_fc(img.to(self.device))
                lidar_embed = self.lidar_fc(lidar.to(self.device))
                txt_embed = self.txt_fc(txt.to(self.device))
                three_embed = torch.stack([img_embed, lidar_embed, txt_embed], dim=0)
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # get gt labels once
                gt_labels = {}
                for idx in range(bs):
                    for jdx in range(idx, bs):
                        gt_labels[idx, jdx] = ds_retrieval_cls.eval_retrieval_ids(
                            self.shuffle2idx[int(orig_idx[idx])],
                            self.shuffle2idx[int(orig_idx[jdx])],
                        )
                # Get one gt_labels = 1 and one gt_labels = 0 for each idx
                positive_pairs = []
                negative_pairs = []
                for idx in range(bs):
                    idx_pos_cnt = 0
                    idx_neg_cnt = 0
                    for jdx in range(idx + 1, bs):
                        if gt_labels[idx, jdx] == 1 and idx_pos_cnt < 1:
                            positive_pairs.append((idx, jdx))
                            idx_pos_cnt += 1
                        elif gt_labels[idx, jdx] == 0 and idx_neg_cnt < 1:
                            negative_pairs.append((idx, jdx))
                            idx_neg_cnt += 1
                        if idx_pos_cnt >= 1 and idx_neg_cnt >= 1:
                            break

                for mod_i in range(3):
                    for mod_j in range(3):
                        for idx, jdx in positive_pairs:
                            if mod_i >= mod_j:
                                continue
                            cos_sim = torch.nn.functional.cosine_similarity(
                                three_embed[mod_i][idx],
                                three_embed[mod_j][jdx],
                                dim=0,
                            )
                            loss = loss + max(0, 1 - cos_sim)
                        for idx, jdx in negative_pairs:
                            loss = loss - max(
                                -torch.nn.functional.cosine_similarity(
                                    three_embed[mod_i][idx],
                                    three_embed[mod_j][jdx],
                                    dim=0,
                                )
                                + 1
                                + 0.4,
                                0,
                            )

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                print(f"Epoch {epoch+1}: loss {loss.item()}")

            if (epoch + 1) % 5 == 0:  # Save models per 5 epochs
                torch.save(
                    self.img_fc.state_dict(),
                    model_path + f"img_fc_epoch_{epoch+1}.pth",
                )
                torch.save(
                    self.lidar_fc.state_dict(),
                    model_path + f"lidar_fc_epoch_{epoch+1}.pth",
                )
                torch.save(
                    self.txt_fc.state_dict(),
                    model_path + f"txt_fc_epoch_{epoch+1}.pth",
                )
                print(f"Models saved at epoch {epoch+1}")

    def load_fc_models(self, epoch: int) -> None:
        """Load the fc models."""
        model_path = self.cfg["KITTI"].paths.save_path + "models/"
        self.define_fc_networks(output_dim=256)
        self.img_fc.load_state_dict(
            torch.load(model_path + f"img_fc_epoch_{epoch}.pth", weights_only=True)
        )
        self.img_fc.to(self.device)
        self.lidar_fc.load_state_dict(
            torch.load(model_path + f"lidar_fc_epoch_{epoch}.pth", weights_only=True)
        )
        self.lidar_fc.to(self.device)
        self.txt_fc.load_state_dict(
            torch.load(model_path + f"txt_fc_epoch_{epoch}.pth", weights_only=True)
        )
        self.txt_fc.to(self.device)

    def transform_with_fc(
        self,
        img: torch.Tensor | np.ndarray,
        lidar: torch.Tensor | np.ndarray,
        txt: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform the data with the fc networks."""
        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        if isinstance(lidar, np.ndarray):
            lidar = torch.tensor(lidar)
        if isinstance(txt, np.ndarray):
            txt = torch.tensor(txt)
        img = img.to(self.device)
        lidar = lidar.to(self.device)
        txt = txt.to(self.device)
        self.img_fc.eval()
        self.lidar_fc.eval()
        self.txt_fc.eval()

        with torch.no_grad():
            img_batches = torch.split(img, 64)
            lidar_batches = torch.split(lidar, 64)
            txt_batches = torch.split(txt, 64)

            img_transformed = []
            lidar_transformed = []
            txt_transformed = []

            for img_batch, lidar_batch, txt_batch in zip(
                img_batches, lidar_batches, txt_batches, strict=True
            ):
                img_transformed.append(self.img_fc(img_batch).cpu().numpy())
                lidar_transformed.append(self.lidar_fc(lidar_batch).cpu().numpy())
                txt_transformed.append(self.txt_fc(txt_batch).cpu().numpy())

            img_transformed = np.concatenate(img_transformed, axis=0)
            lidar_transformed = np.concatenate(lidar_transformed, axis=0)
            txt_transformed = np.concatenate(txt_transformed, axis=0)

        return img_transformed, lidar_transformed, txt_transformed

    def define_fc_networks(self, output_dim: int) -> None:
        """Define the initial three 3-layer fully connected networks with a specified output dimension."""
        self.img_fc = torch.nn.Sequential(
            torch.nn.Linear(self.imgdata["train"].shape[1], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, output_dim),
            torch.nn.ReLU(),
        )

        self.lidar_fc = torch.nn.Sequential(
            torch.nn.Linear(self.lidardata["train"].shape[1], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, output_dim),
            torch.nn.ReLU(),
        )

        self.txt_fc = torch.nn.Sequential(
            torch.nn.Linear(self.txtdata["train"].shape[1], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, output_dim),
            torch.nn.ReLU(),
        )

    def get_joint_dataloader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 4
    ) -> torch.utils.data.DataLoader:
        """Create a PyTorch DataLoader for imgdata, lidardata, and txtdata jointly.

        Args:
            batch_size: The number of samples per batch to load.
            shuffle: Set to True to have the data reshuffled at every epoch.
            num_workers: How many subprocesses to use for data loading.

        Returns:
            DataLoader: A PyTorch DataLoader for the joint dataset.
        """

        class JointDataset(torch.utils.data.Dataset):
            def __init__(
                self,
                imgdata: np.ndarray,
                lidardata: np.ndarray,
                txtdata: np.ndarray,
            ) -> None:
                self.imgdata = imgdata
                self.lidardata = lidardata
                self.txtdata = txtdata

            def __len__(self):  # noqa: ANN204
                return len(self.imgdata)

            def __getitem__(
                self, idx: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                img = self.imgdata[idx]
                lidar = self.lidardata[idx]
                txt = self.txtdata[idx]
                return img, lidar, txt, idx

        joint_dataset = JointDataset(
            torch.tensor(self.imgdata["train"]),
            torch.tensor(self.lidardata["train"]),
            torch.tensor(self.txtdata["train"]),
        )
        return torch.utils.data.DataLoader(
            joint_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def eval_similarity(
        self,
        q_feats: np.ndarray,  # (3, 1, 256)
        r_feats: np.ndarray,  # (3, 1, 256)
        q_missing_modalities: list[int],
        r_missing_modalities: list[int],
    ) -> float:
        """Evaluate the similarity between two data points."""
        sim_score = 0
        cnt = 0
        for q_modality in range(3):
            for r_modality in range(3):
                if (
                    q_modality in q_missing_modalities
                    or r_modality in r_missing_modalities
                ):
                    continue
                cnt += 1
                sim_score += cosine_sim(
                    q_feats[q_modality].reshape(1, -1),
                    r_feats[r_modality].reshape(1, -1),
                )
        if cnt == 0:
            return -1
        return sim_score / cnt

    def retrieve_data(
        self,
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
        recalls = {1: [], 5: [], 20: []}
        precisions = {1: [], 5: [], 20: []}
        maps = {5: [], 20: []}
        ds_retrieval_cls = KITTI_file_Retrieval()

        for ii, idx_q in enumerate(
            tqdm(
                self.test_idx,
                desc="Retrieving data",
                leave=True,
            )
        ):
            ds_idx_q = self.shuffle2idx[idx_q]
            retrieved_pairs = []

            # check which modalities are missing
            q_missing_modalities = []
            for modality in range(3):
                if ds_idx_q in self.mask[modality]:
                    q_missing_modalities.append(modality)
            q_feats = np.stack(
                [
                    self.imgdata["test"][ii].reshape(1, -1),
                    self.lidardata["test"][ii].reshape(1, -1),
                    self.txtdata["test"][ii].reshape(1, -1),
                ],
                axis=0,
            )
            assert q_feats.shape[0:2] == (3, 1), f"{q_feats.shape}"

            for jj, idx_r in enumerate(self.test_idx):
                if idx_r == idx_q:  # cannot retrieve itself
                    continue
                ds_idx_r = self.shuffle2idx[idx_r]
                r_missing_modalities = []
                for modality in range(3):
                    if ds_idx_r in self.mask[modality]:
                        r_missing_modalities.append(modality)
                r_feats = np.stack(
                    [
                        self.imgdata["test"][jj].reshape(1, -1),
                        self.lidardata["test"][jj].reshape(1, -1),
                        self.txtdata["test"][jj].reshape(1, -1),
                    ],
                    axis=0,
                )
                assert r_feats.shape[0:2] == (3, 1), f"{r_feats.shape}"

                gt_label = ds_retrieval_cls.eval_retrieval_ids(ds_idx_q, ds_idx_r)
                sim_score = self.eval_similarity(
                    q_feats,
                    r_feats,
                    q_missing_modalities,
                    r_missing_modalities,
                )

                retrieved_pairs.append((ds_idx_q, ds_idx_r, sim_score, gt_label))

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


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2 poetry run python mmda/baselines/emma_ds_class.py
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/main.yaml")
    ds = KITTIEMMADataset(cfg)
    ds.preprocess_retrieval_data()
    if False:
        ds.train_crossmodal_similarity(max_epoch=100)
    ds.load_fc_models(epoch=100)
    img_transformed, lidar_transformed, txt_transformed = ds.transform_with_fc(
        ds.imgdata["test"], ds.lidardata["test"], ds.txtdata["test"]
    )
    ds.imgdata["test"] = img_transformed
    ds.lidardata["test"] = lidar_transformed
    ds.txtdata["test"] = txt_transformed
    maps, precisions, recalls = ds.retrieve_data()
    print(maps, precisions, recalls)
