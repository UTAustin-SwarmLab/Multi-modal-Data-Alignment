"""Dataset class for EMMA retrieval task."""

# ruff: noqa: S301
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from mmda.utils.dataset_utils import load_msrvtt
from mmda.utils.liploc_model import get_top_k
from mmda.utils.sim_utils import cosine_sim


class MSRVTTEmmaDataset:
    """MSRVTT dataset class for EMMA multimodal retrieval task."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the dataset.

        Args:
            cfg: configuration file
        """
        np.random.seed(0)
        torch.manual_seed(0)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(self.cfg_dataset.paths.save_path) / "models"

        self.txt2img_emb = 0

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
        self.num_data = self.txt2img_emb.shape[0]

        # handle missing audio in videos
        self.audio2txt_emb[self.null_audio_idx] = 0  # 2848 missing out of 24403
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
        # load data
        self.load_data()
        assert (
            self.test_size + self.cali_size + self.train_size == self.num_data
        ), f"{self.test_size} + {self.cali_size} + {self.train_size} != {self.num_data}"

        # train/test/calibration split only on the query size (59_800)
        idx = np.arange(self.num_data)
        self.video_id_to_txt_idx = {}
        self.img_train_size = 0
        self.img_test_size = 0
        cnt = 0
        for q_idx in range(self.num_data):
            video_id = self.video_info_sen_order[q_idx]["video_id"]
            if video_id not in self.video_id_to_txt_idx:
                self.video_id_to_txt_idx[video_id] = []
                cnt += 1
            self.video_id_to_txt_idx[video_id].append(q_idx)
            if q_idx == self.train_size:
                self.img_train_size = cnt
            elif q_idx == self.train_size + self.test_size:
                self.img_test_size = cnt

        img_train_idx = idx[: self.img_train_size]
        img_test_idx = idx[
            self.img_train_size : self.img_train_size + self.img_test_size
        ]
        self.img2txt_emb = {
            "train": self.img2txt_emb[img_train_idx],
            "test": self.img2txt_emb[img_test_idx],
        }
        self.audio2txt_emb = {
            "train": self.audio2txt_emb[img_train_idx],
            "test": self.audio2txt_emb[img_test_idx],
        }
        # masking missing data in the test set. Mask the whole modality of an instance at a time.
        self.mask = {}
        self.mask[1] = []
        if self.cfg_dataset.mask_ratio != 0:
            mask_num = int(self.test_size / self.cfg_dataset.mask_ratio)
            # mask the img modality only since the audio modality already has missing data
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

    def train_crossmodal_similarity(  # noqa: C901, PLR0912
        self, max_epoch: int
    ) -> None:
        """Train the cross-modal similarity, aka the CSA method."""
        data_loader = self.get_joint_dataloader(batch_size=256, num_workers=4)
        self.define_fc_networks(output_dim=256)
        self.img_fc.to(self.device)
        self.audio_fc.to(self.device)
        self.txt_fc.to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.img_fc.parameters())
            + list(self.audio_fc.parameters())
            + list(self.txt_fc.parameters()),
            lr=0.001,
        )

        self.model_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(max_epoch):
            for _, (img, txt, audio, _) in enumerate(data_loader):
                bs = img.shape[0]
                img_embed = self.img_fc(img.to(self.device).float())
                audio_embed = self.audio_fc(audio.to(self.device).float())
                txt_embed = self.txt_fc(txt.to(self.device).float())
                three_embed = torch.stack([img_embed, audio_embed, txt_embed], dim=0)
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # Get one gt_labels = 1 and one gt_labels = 0 for each idx
                positive_pairs = []
                negative_pairs = []
                for idx in range(bs):
                    idx_pos_cnt = 0
                    idx_neg_cnt = 0
                    for jdx in range(idx + 1, bs):
                        if idx == jdx and idx_pos_cnt < 1:
                            positive_pairs.append((idx, jdx))
                            idx_pos_cnt += 1
                        elif idx != jdx and idx_neg_cnt < 1:
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
                    str(self.model_path / f"img_fc_epoch_{epoch+1}{self.save_tag}.pth"),
                )
                torch.save(
                    self.audio_fc.state_dict(),
                    str(
                        self.model_path / f"audio_fc_epoch_{epoch+1}{self.save_tag}.pth"
                    ),
                )
                torch.save(
                    self.txt_fc.state_dict(),
                    str(self.model_path / f"txt_fc_epoch_{epoch+1}{self.save_tag}.pth"),
                )
                print(f"Models saved at epoch {epoch+1}")

    def load_fc_models(self, epoch: int) -> None:
        """Load the fc models."""
        model_path = Path(self.cfg_dataset.paths.save_path) / "models"
        self.define_fc_networks(output_dim=256)
        self.img_fc.load_state_dict(
            torch.load(
                str(model_path / f"img_fc_epoch_{epoch}{self.save_tag}.pth"),
                weights_only=True,
            )
        )
        self.img_fc.to(self.device)
        self.audio_fc.load_state_dict(
            torch.load(
                str(model_path / f"audio_fc_epoch_{epoch}{self.save_tag}.pth"),
                weights_only=True,
            )
        )
        self.audio_fc.to(self.device)
        self.txt_fc.load_state_dict(
            torch.load(
                str(model_path / f"txt_fc_epoch_{epoch}{self.save_tag}.pth"),
                weights_only=True,
            )
        )
        self.txt_fc.to(self.device)

    def transform_with_fc(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform the data with the fc networks."""
        img = np.concatenate(
            [self.img2txt_emb["train"], self.img2txt_emb["test"]], axis=0
        )
        audio = np.concatenate(
            [self.audio2txt_emb["train"], self.audio2txt_emb["test"]], axis=0
        )
        txt = np.concatenate([self.txt2img_emb, self.txt2audio_emb], axis=1)
        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio)
        if isinstance(txt, np.ndarray):
            txt = torch.tensor(txt)
        img = img.to(self.device).float()
        audio = audio.to(self.device).float()
        txt = txt.to(self.device).float()
        self.img_fc.eval()
        self.audio_fc.eval()
        self.txt_fc.eval()

        with torch.no_grad():
            img_batches = torch.split(img, 256)
            audio_batches = torch.split(audio, 256)
            img_transformed = []
            audio_transformed = []

            for img_batch, audio_batch in zip(img_batches, audio_batches, strict=True):
                img_transformed.append(self.img_fc(img_batch).cpu().numpy())
                audio_transformed.append(self.audio_fc(audio_batch).cpu().numpy())

            img_transformed = np.concatenate(img_transformed, axis=0)
            audio_transformed = np.concatenate(audio_transformed, axis=0)

            # tranform text separately as it has a different number of samples
            txt_batches = torch.split(txt, 256)
            txt_transformed = []
            for txt_batch in txt_batches:
                txt_transformed.append(self.txt_fc(txt_batch).cpu().numpy())

            txt_transformed = np.concatenate(txt_transformed, axis=0)

        self.img2txt_emb_all = img_transformed
        self.txt_emb_all = txt_transformed
        self.audio2txt_emb_all = audio_transformed

    def define_fc_networks(self, output_dim: int) -> None:
        """Define the initial three 3-layer fully connected networks with a specified output dimension."""
        self.img_fc = torch.nn.Sequential(
            torch.nn.Linear(self.img2txt_emb["train"].shape[1], 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, output_dim),
            torch.nn.ReLU(),
        )

        self.audio_fc = torch.nn.Sequential(
            torch.nn.Linear(self.audio2txt_emb["train"].shape[1], 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, output_dim),
            torch.nn.ReLU(),
        )

        self.txt_fc = torch.nn.Sequential(
            torch.nn.Linear(
                self.txt2img_emb.shape[1] + self.txt2audio_emb.shape[1],
                2048,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, output_dim),
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
                txt2img_emb: np.ndarray,
                txt2audio_emb: np.ndarray,
                audio2txt_emb: np.ndarray,
                img2txt_emb: np.ndarray,
                video_id_to_txt_idx: list[dict],
                ref_id_order: list[str],
            ) -> None:
                self.txt2img_emb = txt2img_emb
                self.txt2audio_emb = txt2audio_emb
                self.audio2txt_emb = audio2txt_emb
                self.img2txt_emb = img2txt_emb
                self.video_id_to_txt_idx = video_id_to_txt_idx
                self.ref_id_order = ref_id_order

            def __len__(self):  # noqa: ANN204
                return self.audio2txt_emb.shape[0]

            def __getitem__(
                self, idx: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                audio = self.audio2txt_emb[idx]
                img = self.img2txt_emb[idx]
                video_id = self.ref_id_order[idx]
                txt_idx = np.random.choice(self.video_id_to_txt_idx[video_id])
                txt = torch.concat(
                    [self.txt2img_emb[txt_idx], self.txt2audio_emb[txt_idx]], axis=0
                )
                return img, txt, audio, idx

        joint_dataset = JointDataset(
            torch.tensor(self.txt2img_emb),
            torch.tensor(self.txt2audio_emb),
            torch.tensor(self.audio2txt_emb["train"]),
            torch.tensor(self.img2txt_emb["train"]),
            self.video_id_to_txt_idx,
            self.ref_id_order,
        )
        return torch.utils.data.DataLoader(
            joint_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def eval_similarity(
        self,
        q_feats: np.ndarray,  # (1, 1, 256)
        r_feats: np.ndarray,  # (2, 1, 256)
        r_missing_modalities: list[int],
    ) -> float:
        """Evaluate the similarity between two data points."""
        sim_score = 0
        cnt = 0
        for r_modality in range(2):
            if r_modality in r_missing_modalities:
                continue
            cnt += 1
            sim_score += cosine_sim(
                q_feats[0],
                r_feats[r_modality],
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

        for idx_q in tqdm(
            range(self.train_size, self.train_size + self.test_size),
            desc="Retrieving data",
            leave=True,
        ):
            retrieved_pairs = []
            q_feats = self.txt_emb_all[idx_q].reshape(1, 1, -1)

            for idx_r in range(self.img2txt_emb_all.shape[0]):
                r_missing_modalities = []
                for modality in range(2):
                    if idx_r in self.mask[modality]:
                        r_missing_modalities.append(modality)
                r_feats = np.stack(
                    [
                        self.img2txt_emb_all[idx_r].reshape(1, -1),
                        self.audio2txt_emb_all[idx_r].reshape(1, -1),
                    ],
                    axis=0,
                )
                assert r_feats.shape[0:2] == (2, 1), f"{r_feats.shape}"

                gt_label = self.check_correct_retrieval(idx_q, idx_r)
                sim_score = self.eval_similarity(
                    q_feats,
                    r_feats,
                    r_missing_modalities,
                )

                retrieved_pairs.append((idx_q, idx_r, sim_score, gt_label))

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

        maps_dict = {5: np.mean(maps[5]), 20: np.mean(maps[20])}
        precisions_dict = {
            1: np.mean(precisions[1]),
            5: np.mean(precisions[5]),
            20: np.mean(precisions[20]),
        }
        recalls_dict = {
            1: np.mean(recalls[1]),
            5: np.mean(recalls[5]),
            20: np.mean(recalls[20]),
        }
        return maps_dict, precisions_dict, recalls_dict


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2 poetry run python mmda/baselines/emma/emma_msrvtt_class.py
    import pandas as pd
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/main.yaml")
    ds = MSRVTTEmmaDataset(cfg)
    ds.preprocess_retrieval_data()
    if False:
        ds.train_crossmodal_similarity(max_epoch=200)
    ds.load_fc_models(epoch=200)
    ds.transform_with_fc()
    maps, precisions, recalls = ds.retrieve_data()
    print(maps, precisions, recalls)
    # write the results to a csv file
    data = {
        "method": [
            "EMMA",
        ],
        "mAP@5": [maps[5]],
        "mAP@20": [maps[20]],
        "Precision@1": [precisions[1]],
        "Precision@5": [precisions[5]],
        "Precision@20": [precisions[20]],
        "Recall@1": [recalls[1]],
        "Recall@5": [recalls[5]],
        "Recall@20": [recalls[20]],
    }
    df = pd.DataFrame(data)
    dir_path = Path(cfg.MSRVTT.paths.plots_path)
    df_path = (
        dir_path
        / f"emma_msrvtt_class_{cfg.MSRVTT.img_encoder}_{cfg.MSRVTT.audio_encoder}.csv"
    )
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(df_path, index=False)
