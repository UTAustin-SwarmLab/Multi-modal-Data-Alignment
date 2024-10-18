"""Test script for ImageBind."""

import torch
import torchaudio
import wave
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

# from imagebind.data import (
#     load_and_transform_audio_data,
#     load_and_transform_text,
#     load_and_transform_vision_data,
# )
# from imagebind.models import imagebind_model
# from imagebind.models.imagebind_model import ModalityType
from mmda.utils.imagebind_utils import ImageBindInference
from pathlib import Path

import hydra
from mmda.utils.dataset_utils import load_msrvtt
from mmda.utils.imagebind_utils import ImageBindInference
import os
import pickle

BATCH_SIZE = 256


text_list = ["A dog.", "A car", "A bird"]
image_paths = [
    ".assets/dog_image.jpg",
    ".assets/car_image.jpg",
    ".assets/bird_image.jpg",
]
audio_paths = [
    ".assets/dog_audio.wav",
    ".assets/car_audio.wav",
    ".assets/bird_audio.wav",
]
with wave.open(audio_paths[0], "r") as wav_file:
    print(f"Channels: {wav_file.getnchannels()}")
    print(f"Sample width: {wav_file.getsampwidth()}")
    print(f"Frame rate (Sample rate): {wav_file.getframerate()}")
    print(f"Number of frames: {wav_file.getnframes()}")
    print(f"Compression type: {wav_file.getcomptype()}")
waveform, sample_rate = torchaudio.load(audio_paths[0])
print(waveform, sample_rate)

with wave.open("/nas/pohan/datasets/MSR-VTT/TestVideo/video7010.wav", "r") as wav_file:
    print(f"Channels: {wav_file.getnchannels()}")
    print(f"Sample width: {wav_file.getsampwidth()}")
    print(f"Frame rate (Sample rate): {wav_file.getframerate()}")
    print(f"Number of frames: {wav_file.getnframes()}")
    print(f"Compression type: {wav_file.getcomptype()}")
waveform, sample_rate = torchaudio.load(
    "/nas/pohan/datasets/MSR-VTT/TestVideo/video7010.wav"
)
path = Path("/nas/pohan/datasets/MSR-VTT/TestVideo/video7013.wav")
print(path.exists())
print(waveform, sample_rate)
# Check the current backend
backend = torchaudio.get_audio_backend()
print(f"Current torchaudio backend: {backend}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Instantiate model
# model = imagebind_model.imagebind_huge(pretrained=True)
# model.eval()
# model.to(device)

# # Load data
# inputs = {
#     ModalityType.TEXT: load_and_transform_text(text_list, device),
#     ModalityType.VISION: load_and_transform_vision_data(image_paths, device),
#     ModalityType.AUDIO: load_and_transform_audio_data(audio_paths, device),
# }

# with torch.no_grad():
# embeddings = model(inputs)

# print(
#     "Vision x Text: ",
#     torch.softmax(
#         embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1
#     ),
# )
# print(
#     "Audio x Text: ",
#     torch.softmax(
#         embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1
#     ),
# )
# print(
#     "Vision x Audio: ",
#     torch.softmax(
#         embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1
#     ),
# )


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
        for frame_id in range(0, num_frames, 2):
            if frame_id + 1 >= num_frames:
                break
            first_img_path = img_dir / f"{frame_id:04d}.jpg"
            last_img_path = img_dir / f"{frame_id + 1:04d}.jpg"
            id_order.append(video_ids)
            first_img_paths.append(str(first_img_path))
            last_img_paths.append(str(last_img_path))
    return id_order, first_img_paths, last_img_paths


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg):
    cfg_dataset = cfg["MSRVTT"]
    _, captions, video_info_sen_order, video_dict = load_msrvtt(cfg_dataset)

    id_order, first_img_paths, last_img_paths = get_video_emb(
        cfg_dataset, video_dict, use_kaggle=False
    )
    audio_paths = []
    for video_id in id_order:
        audio_path = str(
            Path(cfg_dataset.paths.dataset_path, f"TestVideo/{video_id}.wav")
        )
        audio_paths.append(audio_path)

    imagebind_class = ImageBindInference(0)
    audio_np = []
    for i in range(0, len(id_order), BATCH_SIZE):
        audios = audio_paths[i : i + BATCH_SIZE]
        audio_embs = imagebind_class.inference_audio(audios).cpu().numpy()
        audio_np.append(audio_embs)

    audio_np.append(audio_embs)
    with Path(cfg_dataset.paths.save_path, "MSRVTT_audio_emb_imagebind.pkl").open(
        "wb"
    ) as f:
        pickle.dump(audio_np, f)
    print("imagebind embeddings saved")


if __name__ == "__main__":
    main()
