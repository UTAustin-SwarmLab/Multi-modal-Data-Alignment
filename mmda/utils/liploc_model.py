"""This script is used to evaluate liploc model on the test set."""

# ruff: noqa

import math
from pathlib import Path
import tyro
from dataclasses import dataclass
import numpy as np
import torch
from tqdm.autonotebook import tqdm
import importlib
import pickle
from omegaconf import DictConfig
import hydra


class CFG:
    expid = Path(__file__).stem
    data_path = "/nas/pohan/datasets/KITTI/dataset/sequences"
    data_path_360 = "/nas/pohan/datasets/KITTI-360"
    debug = False
    train_sequences = [
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "11",
        "12",
        "13",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
    ]
    expdir = f"data/{expid}/"
    best_model_path = "/nas/pohan/models/liploc_largest_vit_best.pth"  # modified
    final_model_path = f"{expdir}model.pth"
    batch_size = 512
    num_workers = 4  # 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_image_model_name = "vit_small_patch16_224"  #'resnet50'
    image_embedding_dim = 2048
    max_length = 200
    pretrained = True  # for both image encoder and text encoder
    trainable = True  # for both image encoder and text encoder
    temperature = 1.0
    # image size
    size = 224
    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1
    crop = True
    dataloader = "KittiBothDataset"
    model = "CLIPModelV1_vit"

    logdir = f"data/{expid}/log/"

    details = (
        f"Exp Id: {expid} \nTraining on: {train_sequences} \nBatch Size: {batch_size}"
    )


@dataclass
class Args:
    expid: str = "exp_default"
    eval_sequence = ["04", "05", "06", "07", "08", "09", "10"]
    threshold_dist: int = 5


model_import_path = f"mmda.liploc.models.{CFG.model}"
dataloader_import_path = f"mmda.liploc.dataloaders.{CFG.dataloader}"
model = importlib.import_module(model_import_path).Model(CFG)
get_topk = importlib.import_module(model_import_path).get_topk
get_dataloader = importlib.import_module(dataloader_import_path).get_dataloader
get_filenames = importlib.import_module(dataloader_import_path).get_filenames
get_poses = importlib.import_module(dataloader_import_path).get_poses
args = tyro.cli(Args)


def get_lidar_image_embeddings(filenames, model):
    valid_loader = get_dataloader(filenames, mode="valid", CFG=CFG)

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_embeddings = model.get_lidar_embeddings(batch)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)


def get_camera_image_embeddings(filenames, model):
    valid_loader = get_dataloader(filenames, mode="valid", CFG=CFG)

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_embeddings = model.get_camera_embeddings(batch)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)


def find_matches(ref_embeddings, query_embeddings, image_filenames, top_k=1):
    if isinstance(ref_embeddings, np.ndarray):
        ref_embeddings = torch.from_numpy(ref_embeddings)
    if isinstance(query_embeddings, np.ndarray):
        query_embeddings = torch.from_numpy(query_embeddings)
    values, indices = get_topk(
        torch.unsqueeze(query_embeddings, 0), ref_embeddings, top_k
    )
    matches = [image_filenames[idx] for idx in indices]
    return matches


def load_liploc_model():
    model_path = CFG.best_model_path
    model.to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    return model


def load_eval_filenames():
    all_filenames = np.array([])
    for sequence in args.eval_sequence:
        filenames = get_filenames([sequence], CFG.data_path, CFG.data_path_360)
        # merge all filenames in np.array
        all_filenames = np.append(all_filenames, filenames)
    # print("all_filenames", all_filenames, len(all_filenames))
    return all_filenames


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def get_liploc_embeddings(cfg: DictConfig):
    print("Evaluating On: ", args.eval_sequence)
    cfg_dataset = cfg.KITTI

    model = load_liploc_model()
    all_filenames = load_eval_filenames()

    if len(args.eval_sequence) == 4:  # KITTI360
        translation_poses, indices = get_poses(args.eval_sequence, CFG)
        all_filenames = all_filenames[indices.astype(int)]

    # create dictionary of embeddings
    Path(cfg_dataset.paths.save_path).mkdir(parents=True, exist_ok=True)

    # get embeddings
    print("Getting Lidar Embeddings...")
    lidar_embeddigs = get_lidar_image_embeddings(all_filenames, model).cpu().numpy()
    print(lidar_embeddigs.shape)
    with Path(cfg_dataset.paths.save_path, "KITTI_lidar_emb_liploc.pkl").open(
        "wb"
    ) as f:
        pickle.dump(lidar_embeddigs, f)
        print("lidar embeddings saved")

    print("Getting Camera Embeddings...")
    camera_embeddings = get_camera_image_embeddings(all_filenames, model).cpu().numpy()
    print(camera_embeddings.shape)
    with Path(cfg_dataset.paths.save_path, "KITTI_camera_emb_liploc.pkl").open(
        "wb"
    ) as f:
        pickle.dump(camera_embeddings, f)
        print("camera embeddings saved")
    return


def eval_liploc_query(ref_embeddings, query_embeddings, query_ids, top_k: int = 1):
    """
    query_embeddings: (num_queries, embedding_dim)
    ref_embeddings: (num_references, embedding_dim)
    query_ids: list(num_queries)
    top_k: number of top matches to consider

    num_references should be equal to the number of filenames
    """
    # query_predict = []
    all_filenames = load_eval_filenames()
    # sanity check
    assert (
        all_filenames.size == ref_embeddings.shape[0]
    ), f"Mismatch {all_filenames.size} != {ref_embeddings.size(0)}"
    assert top_k < len(all_filenames), f"top_k greater than the number of files"

    translation_poses = None
    indices = None
    for sequence in args.eval_sequence:
        if len(sequence) == 2:  # KITTI
            translation_pose = get_poses(sequence, CFG)
            # print("Translation Poses: ", translation_poses.shape)  # (271, 3)
        elif len(sequence) == 4:  # KITTI360
            translation_pose, indice = get_poses(sequence, CFG)
            all_filenames = all_filenames[indices.astype(int)]
            # merge dictionary of indices
            if indices is None:
                indices = indice
            else:
                indices.update(indice)
        else:
            raise ValueError("Invalid sequence")
        # concatenate all translation poses
        if translation_poses is None:
            translation_poses = translation_pose
        else:
            translation_poses = np.concatenate(
                (translation_poses, translation_pose), axis=0
            )

    print("Evaluating On: ", args.eval_sequence)
    recalls = []
    precisions = []
    mAPs = []

    # for i, filename in tqdm(enumerate(all_filenames)):
    for i in query_ids:
        filename = all_filenames[i]
        distances = []
        if len(args.eval_sequence[0]) == 2:
            queryimagefilename = filename.split("/")[1]
            predictions = find_matches(  # return filenames
                ref_embeddings=ref_embeddings,
                query_embeddings=query_embeddings[i],
                image_filenames=all_filenames,
                top_k=top_k,
            )
            for prediction in predictions:
                predictedPose = int(prediction.split("/")[1])
                queryPose = int(queryimagefilename)
                # query_predict.append([queryPose, predictedPose])
                # only considers x and y coordinates of a prediction
                distance = math.sqrt(
                    (
                        translation_poses[queryPose][1]
                        - translation_poses[predictedPose][1]
                    )
                    ** 2
                    + (
                        translation_poses[queryPose][2]
                        - translation_poses[predictedPose][2]
                    )
                    ** 2
                )
                distances.append(distance)
        else:
            values, pred_idx = get_topk(
                torch.unsqueeze(query_embeddings[i], 0), ref_embeddings, top_k
            )
            for predIdx in pred_idx:
                queryIdx = i
                distance = math.sqrt(
                    (translation_poses[queryIdx][1] - translation_poses[predIdx][1])
                    ** 2
                    + (translation_poses[queryIdx][2] - translation_poses[predIdx][2])
                    ** 2
                )
                distances.append(distance)

        # Calculate Recall, Precision, mAP
        num_matches = 0
        true_positives = 0
        false_positives = 0
        hit = []
        for distance in distances:
            if distance < args.threshold_dist:
                num_matches += 1
                true_positives += 1
                hit.append(1.0)
            else:
                false_positives += 1
                hit.append(0.0)

        # Recall = True Positives / Actual Positives (num_total here is treated as total positives)
        # Precision = True Positives / (True Positives + False Positives)
        recall = 1 if true_positives > 0 else 0
        precision = true_positives / (true_positives + false_positives)
        # To calculate mAP (mean Average Precision)
        # Precision@k = (Number of relevant items in top-k predictions) / k
        # Average Precision = Σ (Precision@k * (change in recall@k)) / Total relevant items
        # mAP = Σ Average Precision / Total queries
        precision_at_k = np.cumsum(hit) / (np.arange(top_k) + 1)  # array
        ap = np.sum(precision_at_k * hit) / top_k

        recalls.append(recall)
        precisions.append(precision)
        mAPs.append(ap)

    # query_predict = np.array([query_predict])
    # np.save(f"data/eval_predictions_{args.eval_sequence}.np", query_predict)

    # Output the results
    print("=========================================\nTop_k: ", top_k)
    print(f"mAP: {np.array(mAPs).mean()}")
    print(f"Precision: {np.array(precisions).mean()}")
    print(f"Recall: {np.array(recalls).mean()}")

    return np.array(mAPs).mean(), np.array(precisions).mean(), np.array(recalls).mean()


class KITTI_file_Retrieval:
    def __init__(self):
        self.args = tyro.cli(Args)
        self.translation_poses = {}
        indices = {}
        for sequence in self.args.eval_sequence:
            if len(sequence) == 2:  # KITTI
                translation_pose = get_poses(sequence, CFG)
            elif len(sequence) == 4:  # KITTI360
                translation_pose, indice = get_poses(sequence, CFG)
                # merge dictionary of indicess
                indices[sequence] = indice
            else:
                raise ValueError("Invalid sequence")
            # concatenate all translation poses
            self.translation_poses[int(sequence)] = translation_pose

        self.all_filenames = load_eval_filenames()

    def eval_retrieval_ids(self, query_id: int, ref_id: int) -> int:
        """Only for KITTI dataset.

        Args:
            query_id: int
            ref_id: int

        Returns:
            int: 1 if the distance is less than the threshold, 0 otherwise
        """
        queryimagefilename = self.all_filenames[query_id]
        pred_filename = self.all_filenames[ref_id]
        query_sequence = int(queryimagefilename.split("/")[0])
        pred_sequence = int(pred_filename.split("/")[0])
        predictedPose = int(pred_filename.split("/")[1])
        queryPose = int(queryimagefilename.split("/")[1])
        # only considers x and y coordinates of a prediction
        distance = math.sqrt(
            (
                self.translation_poses[query_sequence][queryPose][1]
                - self.translation_poses[pred_sequence][predictedPose][1]
            )
            ** 2
            + (
                self.translation_poses[query_sequence][queryPose][2]
                - self.translation_poses[pred_sequence][predictedPose][2]
            )
            ** 2
        )
        return int(distance < args.threshold_dist)


def get_top_k(retrieved_pairs: list[tuple[int, int, float]], k: int) -> list[bool]:
    """Calculate the top k hit for the test data's similarity matrix.

    Args:
        retrieved_pairs: the retrieved pairs in the format of (idx_1, idx_2, conformal_probability)
            and in descending order of the conformal probability
        k: the number of top k pairs to retrieve

    Returns:
        top_k_hit: the top k hit
    """
    top_k_hit = []
    preds = retrieved_pairs[:k]
    for _, _, _, gt_label in preds:
        if gt_label == 1:
            top_k_hit.append(True)
        else:
            top_k_hit.append(False)
    return top_k_hit


if __name__ == "__main__":
    # get_liploc_embeddings()

    # test eval_liploc_query
    np.random.seed(0)
    ref_emb = np.random.randint(0, 100, size=(12097, 256))
    query_emb = np.random.randint(0, 100, size=(13, 256))
    # ref_emb = np.zeros((12097, 256))
    # query_emb = np.zeros((13, 256))
    # eval_liploc_query(ref_emb, query_emb, query_ids=np.arange(13), top_k=5)
    a = LiplocRetrieval()
    a.eval_retrieval_ids(query_id=0, ref_id=1)

# CUDA_VISIBLE_DEVICES=1 poetry run python ./mmda/utils/liploc_model.py
