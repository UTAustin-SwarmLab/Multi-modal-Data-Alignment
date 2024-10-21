"""Get feature embeddings for the datasets."""

# ruff: noqa: ERA001, PLR2004, S301
import pickle
from pathlib import Path

import numpy as np
import torch
import torchaudio
from moviepy.audio.io.AudioFileClip import AudioFileClip
from omegaconf import DictConfig
from tqdm import tqdm

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
from mmda.utils.imagebind_utils import ImageBindInference
from mmda.utils.video_audio_utils import get_video_emb, prepare_audio_for_imagebind

BATCH_SIZE = 256


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: PLR0915, C901, PLR0912
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
        _, captions, video_info_sen_order, video_dict = load_msrvtt(cfg_dataset)

        id_order, img_paths, audio_start_secs, audio_num_secs = get_video_emb(
            cfg_dataset, video_dict
        )

        # get audio embeddings
        if not (
            Path(cfg_dataset.paths.save_path, "MSRVTT_id_order.pkl").exists
            and Path(cfg_dataset.paths.save_path, "MSRVTT_null_audio.pkl").exists()
            and Path(cfg_dataset.paths.save_path, "MSRVTT_output_paths.pkl").exists()
        ):
            null_audio = []
            output_paths = []
            for i in tqdm(range(len(id_order)), desc="process id_order"):
                video_id = id_order[i]
                audio_path = Path(
                    cfg_dataset.paths.dataset_path, f"TestVideo/{video_id}.wav"
                )
                if not audio_path.exists():
                    null_audio.append(True)
                    waveform = torch.zeros((2, 82025))
                    output_paths.append(".assets/bird_audio.wav")
                    continue
                # load audio
                sample_rate = torchaudio.info(audio_path).sample_rate
                start_frame = int(audio_start_secs[i] * sample_rate)
                num_frames = int(audio_num_secs[i] * sample_rate)
                # print(video_id, start_frame, num_frames, sample_rate)
                waveform, sample_rate = torchaudio.load(
                    str(audio_path),
                    frame_offset=start_frame,
                    num_frames=num_frames,
                )
                if torch.sum(waveform) == 0:
                    null_audio.append(True)
                    output_paths.append(".assets/bird_audio.wav")
                else:
                    output_path = str(audio_path).replace(
                        ".wav", f"_seg_{start_frame}.wav"
                    )
                    null_audio.append(False)
                    output_paths.append(output_path)
                    prepare_audio_for_imagebind(
                        output_path,
                        waveform,
                        sample_rate,
                    )

            with Path(cfg_dataset.paths.save_path, "MSRVTT_id_order.pkl").open(
                "wb"
            ) as f:
                pickle.dump(id_order, f)
            with Path(cfg_dataset.paths.save_path, "MSRVTT_null_audio.pkl").open(
                "wb"
            ) as f:
                pickle.dump(null_audio, f)
            with Path(cfg_dataset.paths.save_path, "MSRVTT_output_paths.pkl").open(
                "wb"
            ) as f:
                pickle.dump(output_paths, f)
        else:
            with Path(cfg_dataset.paths.save_path, "MSRVTT_id_order.pkl").open(
                "rb"
            ) as f:
                id_order = pickle.load(f)
            with Path(cfg_dataset.paths.save_path, "MSRVTT_null_audio.pkl").open(
                "rb"
            ) as f:
                null_audio = pickle.load(f)
            with Path(cfg_dataset.paths.save_path, "MSRVTT_output_paths.pkl").open(
                "rb"
            ) as f:
                output_paths = pickle.load(f)

        # get audio embeddings. first, get np array of audio waveforms
        audio_np = []
        for i in tqdm(range(len(output_paths)), desc="process output_paths"):
            output_path = ".assets/bird_audio.wav" if null_audio[i] else output_paths[i]
            audio = AudioFileClip(output_path)
            waveform = np.array(list(audio.iter_frames()))
            audio.close()
            audio_np.append(waveform)

        audio_np = np.array(audio_np)
        audio_emb = clap_audio(audio_np, batch_size=BATCH_SIZE, max_length_s=120)
        with Path(cfg_dataset.paths.save_path, "MSRVTT_audio_emb_clap.pkl").open(
            "wb"
        ) as f:
            pickle.dump(audio_emb, f)
        print("CLAP embeddings saved")
        return

        # inference imagebind
        imagebind_class = ImageBindInference()
        audio_np = []
        img_np = []
        print(len(id_order))
        for i in tqdm(range(0, len(id_order), BATCH_SIZE), desc="imagebind inference"):
            output_path = output_paths[i : i + BATCH_SIZE]
            images = img_paths[i : i + BATCH_SIZE]
            audio_embs = imagebind_class.inference_audio(output_path).cpu().numpy()
            img_embs = imagebind_class.inference_image(images).cpu().numpy()
            audio_np.append(audio_embs)
            img_np.append(img_embs)
            assert img_embs.shape[1] == 1024, f"img.shape: {img_embs.shape}, {i}"
            assert audio_embs.shape[1] == 1024, f"audio.shape: {audio_embs.shape}, {i}"
        audio_np = np.concatenate(audio_np, axis=0)
        img_np = np.concatenate(img_np, axis=0)
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

        # get text embeddings
        imagebind_class = ImageBindInference()
        text_emb = []
        for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="imagebind txt"):
            text_emb.append(
                imagebind_class.inference_text(captions[i : i + BATCH_SIZE])
                .cpu()
                .numpy()
            )
        text_emb = np.concatenate(text_emb, axis=0)
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
# CUDA_VISIBLE_DEVICES=5 poetry run python mmda/get_embeddings.py
