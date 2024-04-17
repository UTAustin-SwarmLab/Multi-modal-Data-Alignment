"""Utility functions for audio processing."""

import numpy as np
import resampy


def convert_to_mono_channel(audio: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Convert stereo audio to mono channel.

    Args:
        audio: stereo audio. shape: (N, channels)
        normalize: normalize the audio. True by default
    Returns:
        mono audio. Range= [-1, 1]. shape: (N, )
    """
    # convert to numpy array
    audio = np.array(audio).astype(np.float32)
    if normalize:
        # normalize the audio
        audio = audio / np.max(np.abs(audio))
    # if the audio is already mono channel then return the original audio
    # otherwise convert the audio to mono channel
    return audio if len(audio.shape) == 1 else audio.mean(axis=1)


def resample_audio(
    audio: np.ndarray, orig_sr: int, target_sr: int = 48_000
) -> np.ndarray:
    """Resample the audio to the target sample rate.

    Args:
        audio: audio data. shape: (N, )
        orig_sr: original sample rate
        target_sr: target sample rate. 48kHz by default, which is the sample rate of CLAP model
    Returns:
        resampled audio. shape: (N, )
    """
    return resampy.resample(audio, orig_sr, target_sr, axis=0)
