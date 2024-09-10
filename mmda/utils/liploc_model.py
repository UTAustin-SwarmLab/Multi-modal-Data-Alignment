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


class CFG:
    expid = Path(__file__).stem
    data_path = "../data/SEMANTIC-KITTI-DATASET/sequences/"  # TODO: Set path
    debug = False
    train_sequences = ["00", "01", "02", "03", "04", "05", "06", "07"]
    expdir = f"data/{expid}/"
    best_model_path = "/nas/pohan/models/liploc_largest_vit.pth"
    # final_model_path = f"{expdir}model.pth"
    batch_size = 32
    num_workers = 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_image_model_name = "resnet50"
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
    dataloader = "KittiDataset"
    model = "CLIPModelV1"

    logdir = f"data/{expid}/log/"

    details = (
        f"Exp Id: {expid} \nTraining on: {train_sequences} \nBatch Size: {batch_size}"
    )


@dataclass
class Args:
    expid: str = "exp_default"
    eval_sequence: str = "04"
    threshold_dist: int = 5


model_import_path = f"mmda.liploc.models.{CFG.model}"
dataloader_import_path = f"mmda.liploc.dataloaders.{CFG.dataloader}"
model = importlib.import_module(model_import_path).Model(CFG)
get_topk = importlib.import_module(model_import_path).get_topk
get_dataloader = importlib.import_module(dataloader_import_path).get_dataloader
get_filenames = importlib.import_module(dataloader_import_path).get_filenames
get_poses = importlib.import_module(dataloader_import_path).get_poses


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


def find_matches(
    model, lidar_embeddings, query_camera_embeddings, image_filenames, n=1
):
    values, indices = get_topk(
        torch.unsqueeze(query_camera_embeddings, 0), lidar_embeddings, n
    )
    matches = [image_filenames[idx] for idx in indices]
    return matches


def main():
    print(CFG.details)
    args = tyro.cli(Args)
    print("Evaluating On: ", args.eval_sequence)
    model_path = CFG.best_model_path

    model.to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    all_filenames = get_filenames(
        [args.eval_sequence], CFG.data_path, CFG.data_path_360
    )

    if len(args.eval_sequence) == 2:
        translation_poses = get_poses(args.eval_sequence, CFG)
    elif len(args.eval_sequence) == 4:
        translation_poses, indices = get_poses(args.eval_sequence, CFG)
        all_filenames = all_filenames[indices.astype(int)]

    # image_embeddings = get_lidar_image_embeddings([args.eval_sequence], model)
    print("Getting Lidar Embeddings...")
    lidar_embeddings = get_lidar_image_embeddings(all_filenames, model)
    lidar_embeddings = lidar_embeddings.cuda()

    print("Getting Camera Embeddings...")
    camera_embeddings = get_camera_image_embeddings(all_filenames, model)
    camera_embeddings = camera_embeddings.cuda()

    # Evaluation distance metric for Recall@1
    num_matches = 0
    total_queries = all_filenames.size

    # Evaluation distance metric
    diff_sum = []
    # for file in query_filenames:
    # Tqdm for progress bar
    print("Running Evaluation...")
    query_predict = []
    for i, filename in tqdm(enumerate(all_filenames)):

        if len(args.eval_sequence) == 2:
            queryimagefilename = filename.split("/")[1]
            predictions = find_matches(
                model,
                lidar_embeddings=lidar_embeddings,
                query_camera_embeddings=camera_embeddings[i],
                image_filenames=all_filenames,
                n=1,
            )
            predictedPose = int(predictions[0].split("/")[1])
            queryPose = int(queryimagefilename)
            query_predict.append([queryPose, predictedPose])
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
                torch.unsqueeze(camera_embeddings[i], 0), lidar_embeddings, 1
            )
            predIdx = pred_idx[0]
            queryIdx = i
            distance = math.sqrt(
                (translation_poses[queryIdx][1] - translation_poses[predIdx][1]) ** 2
                + (translation_poses[queryIdx][2] - translation_poses[predIdx][2]) ** 2
            )
        # diff_sum.append(distance)
        if distance < args.threshold_dist:
            num_matches += 1

    # print(np.mean(diff_sum))
    recall = num_matches / total_queries
    print("Recall@1: ", recall)
    query_predict = np.array([query_predict])
    np.save(f"data/eval_predictions_{args.eval_sequence}.np", query_predict)


if __name__ == "__main__":
    main()
