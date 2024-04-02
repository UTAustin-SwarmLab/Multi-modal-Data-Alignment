# Load model directly
import datasets
import numpy as np
import resampy
import soundfile as sf
import torch
import wavfile
import wget
from transformers import (
    AutoTokenizer,
    ClapModel,
)
from transformers.models.clap import ClapFeatureExtractor
from transformers.models.clap.processing_clap import ClapProcessor


def test_pipeline_and_feature_extractor():
    """Test the pipeline and feature extractor of CLAP model."""
    dataset = datasets.load_dataset("ashraq/esc50")
    audio = dataset["train"]["audio"][0]["array"]
    print(audio.shape, type(audio), max(audio), min(audio))
    sample_rate = 48000

    dataset = datasets.load_from_disk("/nas/pohan/datasets/WikiMuTe")
    audio_url = dataset["train"][0]["audio_url"]
    print(audio_url)

    audio_file = wget.download(audio_url)
    print(audio_file)
    audio, sample_rate = sf.read(audio_file)
    print(audio.shape, sample_rate)
    print(audio[:10])
    audio = (audio[:, 0] + audio[:, 1]) / 2
    # audio = audio / np.max(np.abs(audio))
    audio = audio.reshape(-1, 1)
    print(audio.shape)
    wavfile.write("24k.wav", sample_rate=sample_rate, audio_data=audio)
    # upsample to 32->48kHz
    audio = resampy.resample(audio, sample_rate, 48000, axis=0)
    audio = audio.reshape(-1) / np.max(np.abs(audio))
    print(audio.shape, max(audio), min(audio))

    sample_rate = 48000
    model = ClapModel.from_pretrained("laion/larger_clap_general")
    processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    input_text = [
        "Sound of a dog",
        "Sound of vaccum cleaner",
        'sound as "monkey punk"',
        "The song has been described as a funk, disco and contemporary R&B track, heavily influenced by hip hop",
        "The song features several layers of funk synthesizers in its instrumentation",
    ]
    inputs = processor(text=input_text, audios=audio, return_tensors="pt", padding=True, sampling_rate=sample_rate)
    outputs = model(**inputs)
    logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
    probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
    print(probs)

    model = ClapModel.from_pretrained("laion/larger_clap_general")
    feature_extractor = ClapFeatureExtractor.from_pretrained("laion/larger_clap_general")
    tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_general")
    inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=sample_rate, padding=True, max_length_s=60)
    tokens = tokenizer(input_text, padding=True, return_tensors="pt")
    audio_features = model.get_audio_features(**inputs)
    text_features = model.get_text_features(**tokens)
    # calculate the similarity score
    # cosine similarity as logits
    logit_scale_text = model.logit_scale_t.exp()
    logit_scale_audio = model.logit_scale_a.exp()
    _ = torch.matmul(text_features, audio_features.t()) * logit_scale_text
    logits_per_audio = torch.matmul(audio_features, text_features.t()) * logit_scale_audio
    print(logits_per_audio.softmax(dim=-1))
    assert torch.allclose(
        text_features, outputs.text_embeds, atol=1e-4
    ), f"{text_features[0, :5]} != {outputs.text_embeds[0, :5]}"
    assert torch.allclose(
        audio_features, outputs.audio_embeds, atol=1e-4
    ), f"{audio_features[0, :5]} != {outputs.audio_embeds[0, :5]}"


if __name__ == "__main__":
    test_pipeline_and_feature_extractor()
