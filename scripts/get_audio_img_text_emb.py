import os
import pickle

from omegaconf import DictConfig

from mmda.utils.data_utils import (
    load_MusicCaps,
    load_SOP,
)
from mmda.utils.embed_data import clap_audio, clap_text, clip_imgs, clip_text, dinov2, gtr_text
from mmda.utils.hydra_utils import hydra_main

BATCH_SIZE = 64


@hydra_main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):  # noqa: D103
    dataset = cfg.dataset
    print(f"Dataset: {dataset}")
    if dataset == "sop":
        cfg_dataset = cfg.sop
        img_files, text_descriptions = load_SOP(cfg_dataset)
        # get text embeddings
        print("Loading CLIP")
        text_emb = clip_text(text_descriptions, BATCH_SIZE)  # batched np array
        print("CLIP embeddings done:", text_emb.shape)
        with open(cfg_dataset.paths.save_path + "data/SOP_text_emb_clip.pkl", "wb") as f:
            pickle.dump(text_emb, f)

        print("Loading GTR")
        text_emb = gtr_text(text_descriptions, BATCH_SIZE)  # batched np array
        print("GTR embeddings done:", text_emb.shape)
        with open(cfg_dataset.paths.save_path + "data/SOP_text_emb_gtr.pkl", "wb") as f:
            pickle.dump(text_emb, f)

        # get img embeddings
        img_emb = dinov2(img_files, BATCH_SIZE)  # batched np array
        with open(cfg_dataset.paths.save_path + "data/SOP_img_emb_dino.pkl", "wb") as f:
            pickle.dump(img_emb, f)
        print("DINO embeddings saved")

        img_emb = clip_imgs(img_files, BATCH_SIZE)  # batched np array
        with open(cfg_dataset.paths.save_path + "data/SOP_img_emb_clip.pkl", "wb") as f:
            pickle.dump(img_emb, f)
        print("CLIP embeddings saved")

    elif dataset == "musiccaps":
        cfg_dataset = cfg.musiccaps
        dataframe = load_MusicCaps(cfg_dataset)
        audio_list = dataframe["audio_path"].tolist()
        caption_list = dataframe["caption"].tolist()
        print(f"Number of audio files: {len(audio_list)}. Number of captions: {len(caption_list)}")

        if not os.path.exists(cfg_dataset.paths.save_path):
            os.makedirs(cfg_dataset.paths.save_path)

        clap_text_features = clap_text(caption_list, batch_size=BATCH_SIZE)
        print(clap_text_features.shape)
        with open(cfg_dataset.paths.save_path + "MusicCaps_text_emb_clap.pkl", "wb") as f:
            pickle.dump(clap_text_features, f)

        clip_text_features = clip_text(caption_list, batch_size=BATCH_SIZE)
        print(clip_text_features.shape)
        with open(cfg_dataset.paths.save_path + "MusicCaps_text_emb_clip.pkl", "wb") as f:
            pickle.dump(clip_text_features, f)

        gtr_text_features = gtr_text(caption_list)
        print(gtr_text_features.shape)
        with open(cfg_dataset.paths.save_path + "MusicCaps_text_emb_gtr.pkl", "wb") as f:
            pickle.dump(gtr_text_features, f)

        clap_audio_features = clap_audio(audio_list, batch_size=BATCH_SIZE)
        print(clap_audio_features.shape)
        with open(cfg_dataset.paths.save_path + "MusicCaps_audio_emb_clap.pkl", "wb") as f:
            pickle.dump(clap_audio_features, f)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")


if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=0 poetry run python get_audio_img_text_emb.py
