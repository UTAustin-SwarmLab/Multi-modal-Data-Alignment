import os
import pickle

from omegaconf import DictConfig

import hydra
from mmda.utils.dataset_utils import (
    load_COSMOS,
    load_ImageNet,
    load_MusicCaps,
    load_PITTS,
    load_SOP,
    load_TIIL,
)
from mmda.utils.embed_data import clap_audio, clap_text, clip_imgs, clip_text, cosplace_img, dinov2, gtr_text

BATCH_SIZE = 128


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):  # noqa: D103
    dataset = cfg.dataset
    print(f"Dataset: {dataset}")
    if dataset == "sop":
        cfg_dataset = cfg.sop
        img_files, text_descriptions = load_SOP(cfg_dataset)

        os.makedirs(cfg_dataset.paths.save_path, exist_ok=True)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)  # batched np array
        print("CLIP embeddings done:", text_emb.shape)
        with open(os.path.join(cfg_dataset.paths.save_path, "data/SOP_text_emb_clip.pkl"), "wb") as f:
            pickle.dump(text_emb, f)

        text_emb = gtr_text(text_descriptions)  # batched np array
        print("GTR embeddings done:", text_emb.shape)
        with open(os.path.join(cfg_dataset.paths.save_path, "data/SOP_text_emb_gtr.pkl"), "wb") as f:
            pickle.dump(text_emb, f)

        # get img embeddings
        img_emb = dinov2(img_files, BATCH_SIZE)  # batched np array
        with open(os.path.join(cfg_dataset.paths.save_path, "data/SOP_img_emb_dino.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_files, BATCH_SIZE)  # batched np array
        with open(os.path.join(cfg_dataset.paths.save_path, "data/SOP_img_emb_clip.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "musiccaps":
        cfg_dataset = cfg.musiccaps
        dataframe = load_MusicCaps(cfg_dataset)
        audio_list = dataframe["audio_path"].tolist()
        caption_list = dataframe["caption"].tolist()
        print(f"Number of audio files: {len(audio_list)}. Number of captions: {len(caption_list)}")

        os.makedirs(cfg_dataset.paths.save_path, exist_ok=True)

        clap_text_features = clap_text(caption_list, batch_size=BATCH_SIZE)
        print(clap_text_features.shape)
        with open(os.path.join(cfg_dataset.paths.save_path, "MusicCaps_text_emb_clap.pkl"), "wb") as f:
            pickle.dump(clap_text_features, f)

        clip_text_features = clip_text(caption_list, batch_size=BATCH_SIZE)
        print(clip_text_features.shape)
        with open(os.path.join(cfg_dataset.paths.save_path, "MusicCaps_text_emb_clip.pkl"), "wb") as f:
            pickle.dump(clip_text_features, f)

        gtr_text_features = gtr_text(caption_list)
        print(gtr_text_features.shape)
        with open(os.path.join(cfg_dataset.paths.save_path, "MusicCaps_text_emb_gtr.pkl"), "wb") as f:
            pickle.dump(gtr_text_features, f)

        clap_audio_features = clap_audio(audio_list, batch_size=BATCH_SIZE)
        print(clap_audio_features.shape)
        with open(os.path.join(cfg_dataset.paths.save_path, "MusicCaps_audio_emb_clap.pkl"), "wb") as f:
            pickle.dump(clap_audio_features, f)

    elif dataset == "imagenet":
        cfg_dataset = cfg.imagenet
        img_path, mturks_idx, orig_idx, clsidx_to_labels = load_ImageNet(cfg_dataset)
        orig_labels = [clsidx_to_labels[i] for i in orig_idx]
        print(f"Number of images: {len(img_path)}. Number of labels: {len(orig_labels)}")
        text_descriptions = ["An image of " + label + "." for label in orig_labels]

        os.makedirs(cfg_dataset.paths.save_path, exist_ok=True)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "ImageNet_text_emb_clip.pkl"), "wb") as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text_descriptions)
        with open(os.path.join(cfg_dataset.paths.save_path, "ImageNet_text_emb_gtr.pkl"), "wb") as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(img_path, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "ImageNet_img_emb_dino.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_path, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "ImageNet_img_emb_clip.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "tiil":
        cfg_dataset = cfg.tiil
        img_files, text_descriptions, _, _ = load_TIIL(cfg_dataset)

        os.makedirs(cfg_dataset.paths.save_path, exist_ok=True)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "TIIL_text_emb_clip.pkl"), "wb") as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text_descriptions)
        with open(os.path.join(cfg_dataset.paths.save_path, "TIIL_text_emb_gtr.pkl"), "wb") as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(img_files, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "TIIL_img_emb_dino.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_files, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "TIIL_img_emb_clip.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "cosmos":
        cfg_dataset = cfg.cosmos
        img_files, text_descriptions, _, _ = load_COSMOS(cfg_dataset)

        os.makedirs(cfg_dataset.paths.save_path, exist_ok=True)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "COSMOS_text_emb_clip.pkl"), "wb") as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text_descriptions)
        with open(os.path.join(cfg_dataset.paths.save_path, "COSMOS_text_emb_gtr.pkl"), "wb") as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(img_files, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "COSMOS_img_emb_dino.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_files, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "COSMOS_img_emb_clip.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "pitts":
        cfg_dataset = cfg.pitts
        img_files, text_descriptions, _ = load_PITTS(cfg_dataset)

        os.makedirs(cfg_dataset.paths.save_path, exist_ok=True)

        # get text embeddings
        text_emb = clip_text(text_descriptions, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "PITTS_text_emb_clip.pkl"), "wb") as f:
            pickle.dump(text_emb, f)
        print("CLIP embeddings saved")

        text_emb = gtr_text(text_descriptions)
        with open(os.path.join(cfg_dataset.paths.save_path, "PITTS_text_emb_gtr.pkl"), "wb") as f:
            pickle.dump(text_emb, f)
        print("GTR embeddings saved")

        # get img embeddings
        img_emb = dinov2(img_files, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "PITTS_img_emb_dino.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_files, BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "PITTS_img_emb_clip.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

        img_emb = cosplace_img(img_files, batch_size=BATCH_SIZE)
        with open(os.path.join(cfg_dataset.paths.save_path, "PITTS_img_emb_cosplace.pkl"), "wb") as f:
            pickle.dump(img_emb, f)
        print("Cosplace embeddings saved")
    # TODO: add more datasets
    else:
        raise ValueError(f"Dataset {dataset} not supported.")


if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=0 poetry run python scripts/get_embeddings.py
