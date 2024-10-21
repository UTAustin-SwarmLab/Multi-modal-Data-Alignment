"""Video and audio utils."""

# ruff: noqa: ERA001, PLR2004, S301
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from moviepy.video.io.VideoFileClip import VideoFileClip
from omegaconf import DictConfig
from tqdm import tqdm


def prepare_audio_for_imagebind(
    output_path: str, waveform: torch.Tensor, sample_rate: int
) -> None:
    """Prepare audio for ImageBind.

    Args:
        output_path: path to save the audio
        waveform: waveform of the audio
        sample_rate: sample rate of the audio
    """
    # Extract 2-second segment
    end_frame = int(sample_rate * 2)
    segment = waveform[:, :end_frame]
    # Pad if necessary
    if segment.shape[1] < sample_rate * 2:
        segment = torch.nn.functional.pad(
            segment, (0, sample_rate * 2 - segment.shape[1])
        )
    # Save segment to wav file
    torchaudio.save(output_path, segment, sample_rate)


def get_video_emb(
    cfg_dataset: DictConfig, video_dict: dict
) -> tuple[list[str], list[str], list[str]]:
    """Get video embeddings for the videos in the video_dict.

    Args:
        cfg_dataset: configuration file
        video_dict: a dict of video information
        use_kaggle: whether to use the kaggle dataset

    Returns:
        id_order: list of video ids
        img_paths: list of image paths
        audio_start_secs: list of audio start seconds
        audio_num_secs: list of audio number of seconds
    """
    id_order = []
    img_paths = []
    step_size = 4
    audio_start_secs = []
    audio_num_secs = []
    for video_id in tqdm(sorted(video_dict), desc="loading keyframe paths"):
        # video_ids from 7010 to 7990
        img_dir = Path(cfg_dataset.paths.dataset_path, "keyframes", video_id)
        csv_path = Path(
            cfg_dataset.paths.dataset_path, "csv_mapping", f"{video_id}.csv"
        )
        df = pd.read_csv(str(csv_path))
        num_frames = len(os.listdir(img_dir))
        assert num_frames == len(df), f"num_frames: {num_frames}, len(df): {len(df)}"
        for frame_id in range(0, num_frames - 1, step_size):
            img_path = img_dir / f"{frame_id:04d}.jpg"
            id_order.append(video_id)
            img_paths.append(str(img_path))
            duration = (
                df.iloc[min(frame_id + step_size, num_frames - 1)]["pts_time"]
                - df.iloc[frame_id]["pts_time"]
            )
            assert duration > 0, f"{video_id}, {frame_id}, {duration}"
            audio_start_secs.append(df.iloc[frame_id]["pts_time"])
            audio_num_secs.append(duration)
    return id_order, img_paths, audio_start_secs, audio_num_secs


def process_video_ids(inputs: tuple[DictConfig, list[int]]) -> tuple[int, np.ndarray]:
    """Multithread to process the videos."""
    cfg_dataset, list_ids = inputs
    result_tuple = []
    for video_id in tqdm(list_ids):
        mp4_file = str(
            Path(cfg_dataset.paths.dataset_path, f"TestVideo/{video_id}.mp4")
        )
        audio_exist, audio = extract_audio_from_video(mp4_file)
        audio = (audio[:, 0] + audio[:, 1]) / 2 if audio_exist else None
        result_tuple.append((video_id, audio))
    return result_tuple


def extract_audio_from_video(mp4_file: str) -> tuple[bool, np.ndarray | None]:
    """Extract audio from a video file.

    Args:
        mp4_file: path to the video file

    Returns:
        True if the audio is extracted successfully, False otherwise
    """
    # Extract the audio from the entire video
    video = VideoFileClip(mp4_file)
    audio = video.audio
    if audio is None:
        return False, None
    # Extract the audio as a list of samples
    audio_samples = list(audio.iter_frames())
    # Write the audio to a WAV file
    audio.write_audiofile(mp4_file.replace(".mp4", ".wav"))
    # Convert the list of samples to a NumPy array
    sound_array = np.array(audio_samples)
    return True, sound_array
