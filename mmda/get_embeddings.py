"""Get feature embeddings for the datasets."""

# ruff: noqa: ERA001
import os
import pickle
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from mmda.utils.imagebind_utils import ImageBindInference
import hydra
from mmda.utils.dataset_utils import (
    load_cosmos,
    load_flickr,
    load_imagenet,
    load_kitti,
    load_leafy_spurge,
    load_msrvtt,
    load_musiccaps,
    load_pitts,
    load_sop,
    load_tiil,
)
from mmda.utils.embed_data import (
    clap_audio,
    clap_text,
    clip_imgs,
    clip_text,
    cosplace_img,
    dinov2,
    gtr_text,
)
BATCH_SIZE = 128


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: PLR0915, C901
    """Get feature embeddings for the datasets.

    Args:
        cfg (DictConfig): Configurations.
    """
    dataset = cfg.dataset
    cfg_dataset = cfg[cfg.dataset]
    print(f"Dataset: {dataset}")
    Path(cfg_dataset.paths.save_path).mkdir(parents=True, exist_ok=True)

    if dataset == "musiccaps":
        dataframe = load_musiccaps(cfg_dataset)
        audio_list = dataframe["audio_path"].tolist()
        caption_list = dataframe["caption"].tolist()
        print(
            f"Number of audio files: {len(audio_list)}. Number of captions: {len(caption_list)}"
        )

        clap_text_features = clap_text(caption_list, batch_size=BATCH_SIZE)
        print(clap_text_features.shape)
        with Path(cfg_dataset.paths.save_path, "MusicCaps_text_emb_clap.pkl").open(
            "wb"
        ) as f:
            pickle.dump(clap_text_features, f)

        clip_text_features = clip_text(caption_list, batch_size=BATCH_SIZE)
        print(clip_text_features.shape)
        with Path(cfg_dataset.paths.save_path, "MusicCaps_text_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(clip_text_features, f)

        gtr_text_features = gtr_text(caption_list)
        print(gtr_text_features.shape)
        with Path(cfg_dataset.paths.save_path, "MusicCaps_text_emb_gtr.pkl").open(
            "wb"
        ) as f:
            pickle.dump(gtr_text_features, f)

        clap_audio_features = clap_audio(audio_list, batch_size=BATCH_SIZE)
        print(clap_audio_features.shape)
        with Path(cfg_dataset.paths.save_path, "MusicCaps_audio_emb_clap.pkl").open(
            "wb"
        ) as f:
            pickle.dump(clap_audio_features, f)

    elif dataset == "MSRVTT":

        def get_video_emb(
            cfg_dataset: DictConfig, video_dict: dict, use_kaggle: bool = False
        ) -> dict[str, np.ndarray]:
            """Get video embeddings for the videos in the video_dict.

            Args:
                cfg_dataset: configuration file
                video_dict: a dict of video information
                use_kaggle: whether to use the kaggle dataset

            Returns:
                video embeddings. dict: video_id -> video_embedding (if use_kaggle)
                img_paths. list: list of image paths (if not use_kaggle)
            """
            # skip image embeddings (CLIP is already done from the dataset)
            # load the existing embeddings
            if use_kaggle:
                video_emb = {}
                for video_ids in tqdm(video_dict, desc="Loading video embeddings"):
                    video_np_path = Path(
                        cfg_dataset.paths.dataset_path,
                        f"clip-features-vit-h14/{video_ids}.npy",
                    )
                    # only sample the first and last frame
                    video_np = np.load(video_np_path)[[0, -1], :].reshape(1, -1)
                    video_emb[video_ids] = video_np
                return video_emb
            id_order = []
            first_img_paths = []
            last_img_paths = []
            for video_ids in tqdm(sorted(video_dict), desc="loading keyframe paths"):
                # video_ids from 7010 to 7990
                img_dir = Path(cfg_dataset.paths.dataset_path, "keyframes", video_ids)
                num_frames = len(os.listdir(img_dir))
                for frame_id in range(num_frames, 2):
                    if frame_id + 1 >= num_frames:
                        break
                    first_img_path = img_dir / f"{frame_id:04d}.jpg"
                    last_img_path = img_dir / f"{frame_id + 1:04d}.jpg"
                    id_order.append(video_ids)
                    first_img_paths.append(str(first_img_path))
                    last_img_paths.append(str(last_img_path))
            return id_order, first_img_paths, last_img_paths

        _, captions, video_info_sen_order, video_dict = load_msrvtt(cfg_dataset)

        id_order, first_img_paths, last_img_paths = get_video_emb(
            cfg_dataset, video_dict, use_kaggle=False
        )
        first_img_emb = clip_imgs(first_img_paths, BATCH_SIZE)
        last_img_emb = clip_imgs(last_img_paths, BATCH_SIZE)
        video_id_emb = np.concatenate([first_img_emb, last_img_emb], axis=1)
        with Path(cfg_dataset.paths.save_path, "MSRVTT_video_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(video_id_emb, f)
        print("CLIP embeddings saved")
        with Path(cfg_dataset.paths.save_path, "MSRVTT_ref_video_ids.pkl").open(
            "wb"
        ) as f:
            pickle.dump(id_order, f)

        # get audio embeddings
        audio_paths = []
        for video_id in id_order:
            audio_path = str(Path(cfg_dataset.paths.dataset_path, f"TestVideo/{video_id}.wav"))
            audio_paths.append(audio_path)
        # inference imagebind
        imagebind_class = ImageBindInference(device=0)
        audio_np = []
        img_np = []
        for i in range(0, len(id_order)), BATCH_SIZE:
            audios = audio_paths[i : i + BATCH_SIZE]
            first_images = first_img_paths[i : i + BATCH_SIZE]
            last_images =last_img_paths[i : i + BATCH_SIZE]
            audio_embs = imagebind_class.inference_audio(audios).cpu().numpy()
            first_embs = imagebind_class.inference_image_only(first_images).cpu().numpy()
            last_embs = imagebind_class.inference_image_only(last_images).cpu().numpy()
            img_embs = np.concatenate([first_embs, last_embs], axis=1)
            audio_np.append(audio_embs)
            img_np.append(img_embs)
        audio_np = np.array(audio_np)
        img_np = np.array(img_np)
        with Path(cfg_dataset.paths.save_path, "MSRVTT_audio_emb_imagebind.pkl").open(
            "wb"
        ) as f:
            pickle.dump(audio_np, f)
        print("imagebind embeddings saved")
        with Path(cfg_dataset.paths.save_path, "MSRVTT_video_emb_imagebind.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_np, f)
        print("imagebind embeddings saved")

        shape = video_info_sen_order[0]["audio_np"].shape
        audio_np = [video_dict[video_id]["audio_np"] if video_dict[video_id]["audio_np"] is not None else np.zeros(shape) for video_id in id_order]
        audio_emb = clap_audio(audio_np, batch_size=BATCH_SIZE, max_length_s=60)
        with Path(cfg_dataset.paths.save_path, "MSRVTT_audio_emb_clap.pkl").open(
            "wb"
        ) as f:
            pickle.dump(audio_emb, f)
        print("CLAP embeddings saved")

        # get text embeddings
        imagebind_class.inference_text(captions)
        with Path(cfg_dataset.paths.save_path, "MSRVTT_text_emb_imagebind.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("imagebind embeddings saved")

        text_emb = clip_text(captions, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "MSRVTT_text_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(captions)
        with Path(cfg_dataset.paths.save_path, "MSRVTT_text_emb_gtr.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        text_emb = clap_text(captions, batch_size=BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "MSRVTT_text_emb_clap.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("CLAP embeddings saved")

    elif dataset == "leafy_spurge":
        images, labels, idx2label = load_leafy_spurge(cfg_dataset)
        text_descriptions = [
            "An image of leafy spurge.",
            "An image of anything other than leafy spurge.",
        ]
        text = [text_descriptions[i] for i in labels]

        # get text embeddings
        text_emb = clip_text(text, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "LeafySpurge_text_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text)
        with Path(cfg_dataset.paths.save_path, "LeafySpurge_text_emb_gtr.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(images, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "LeafySpurge_img_emb_dino.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(images, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "LeafySpurge_img_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "cosmos":
        img_files, text_descriptions, _, _ = load_cosmos(cfg_dataset)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "COSMOS_text_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text_descriptions)
        with Path(cfg_dataset.paths.save_path, "COSMOS_text_emb_gtr.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(img_files, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "COSMOS_img_emb_dino.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_files, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "COSMOS_img_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "sop":
        img_files, text_descriptions = load_sop(cfg_dataset)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)  # batched np array
        print("CLIP embeddings done:", text_emb.shape)
        with Path(cfg_dataset.paths.save_path, "data/SOP_text_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)

        text_emb = gtr_text(text_descriptions)  # batched np array
        print("GTR embeddings done:", text_emb.shape)
        with Path(cfg_dataset.paths.save_path, "data/SOP_text_emb_gtr.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)

        # get img embeddings
        img_emb = dinov2(img_files, BATCH_SIZE)  # batched np array
        with Path(cfg_dataset.paths.save_path, "data/SOP_img_emb_dino.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_files, BATCH_SIZE)  # batched np array
        with Path(cfg_dataset.paths.save_path, "data/SOP_img_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "imagenet":
        img_path, mturks_idx, orig_idx, clsidx_to_labels = load_imagenet(cfg_dataset)
        orig_labels = [clsidx_to_labels[i] for i in orig_idx]
        text_descriptions = ["An image of " + label + "." for label in orig_labels]

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "ImageNet_text_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text_descriptions)
        with Path(cfg_dataset.paths.save_path, "ImageNet_text_emb_gtr.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(img_path, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "ImageNet_img_emb_dino.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_path, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "ImageNet_img_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "tiil":
        img_files, text_descriptions, _, _ = load_tiil(cfg_dataset)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "TIIL_text_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text_descriptions)
        with Path(cfg_dataset.paths.save_path, "TIIL_text_emb_gtr.pkl").open("wb") as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(img_files, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "TIIL_img_emb_dino.pkl").open("wb") as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_files, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "TIIL_img_emb_clip.pkl").open("wb") as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "pitts":
        img_files, text_descriptions, _ = load_pitts(cfg_dataset)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "PITTS_text_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text_descriptions)
        with Path(cfg_dataset.paths.save_path, "PITTS_text_emb_gtr.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(img_files, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "PITTS_img_emb_dino.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_files, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "PITTS_img_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

        img_emb = cosplace_img(img_files, batch_size=BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "PITTS_img_emb_cosplace.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("Cosplace embeddings saved")

    elif dataset == "flickr":
        img_files, text_descriptions, _, _ = load_flickr(cfg_dataset)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "Flickr_text_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text_descriptions)
        with Path(cfg_dataset.paths.save_path, "Flickr_text_emb_gtr.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(img_files, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "Flickr_img_emb_dino.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_files, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "Flickr_img_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "KITTI":
        img_paths, lidar_paths, text_descriptions = load_kitti(cfg_dataset)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)
        with Path(cfg_dataset.paths.save_path, "KITTI_text_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text_descriptions)
        with Path(cfg_dataset.paths.save_path, "KITTI_text_emb_gtr.pkl").open(
            "wb"
        ) as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(img_paths, BATCH_SIZE)
        print(img_emb.shape)
        with Path(cfg_dataset.paths.save_path, "KITTI_camera_emb_dino.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_paths, BATCH_SIZE)
        print(img_emb.shape)
        with Path(cfg_dataset.paths.save_path, "KITTI_camera_emb_clip.pkl").open(
            "wb"
        ) as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    # TODO: add more datasets
    else:
        msg = f"Dataset {dataset} not supported."
        raise ValueError(msg)


if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=0 poetry run python mmda/get_embeddings.py
