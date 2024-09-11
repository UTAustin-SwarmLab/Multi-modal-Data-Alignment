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


def find_matches(lidar_embeddings, query_camera_embeddings, image_filenames, n=1):
    values, indices = get_topk(
        torch.unsqueeze(query_camera_embeddings, 0), lidar_embeddings, n
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
    print("all_filenames", all_filenames, len(all_filenames))
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


def eval_liploc_query(ref_embeddings, query_embeddings, top_k: int = 1):
    # query_predict = []
    num_matches = 0
    total_queries = all_filenames.size
    all_filenames = load_eval_filenames(args, CFG)

    if len(args.eval_sequence) == 2:  # KITTI
        translation_poses = get_poses(args.eval_sequence, CFG)
    elif len(args.eval_sequence) == 4:  # KITTI360
        translation_poses, indices = get_poses(args.eval_sequence, CFG)
        all_filenames = all_filenames[indices.astype(int)]
    for i, filename in tqdm(enumerate(all_filenames)):
        assert all_filenames.size == query_embeddings.size(
            0
        ), f"Mismatch {all_filenames.size} != {query_embeddings.size(0)}"
        assert all_filenames.size == ref_embeddings.size(
            0
        ), f"Mismatch {all_filenames.size} != {ref_embeddings.size(0)}"

        if len(args.eval_sequence) == 2:
            queryimagefilename = filename.split("/")[1]
            predictions = find_matches(
                model,
                lidar_embeddings=ref_embeddings,
                query_camera_embeddings=query_embeddings[i],
                image_filenames=all_filenames,
                n=top_k,
            )
            predictedPose = int(predictions[0].split("/")[1])
            queryPose = int(queryimagefilename)
            # query_predict.append([queryPose, predictedPose])
            # only considers x and y coordinates of a prediction
            distance = math.sqrt(
                (translation_poses[queryPose][1] - translation_poses[predictedPose][1])
                ** 2
                + (
                    translation_poses[queryPose][2]
                    - translation_poses[predictedPose][2]
                )
                ** 2
            )
        else:
            values, pred_idx = get_topk(
                torch.unsqueeze(query_embeddings[i], 0), ref_embeddings, 1
            )
            predIdx = pred_idx[0]
            queryIdx = i
            distance = math.sqrt(
                (translation_poses[queryIdx][1] - translation_poses[predIdx][1]) ** 2
                + (translation_poses[queryIdx][2] - translation_poses[predIdx][2]) ** 2
            )
        if distance < args.threshold_dist:
            num_matches += 1

    recall = num_matches / total_queries
    print("Recall@1: ", recall)
    # query_predict = np.array([query_predict])
    # np.save(f"data/eval_predictions_{args.eval_sequence}.np", query_predict)


if __name__ == "__main__":
    get_liploc_embeddings()

# CUDA_VISIBLE_DEVICES=1 poetry run python ./mmda/utils/liploc_model.py
