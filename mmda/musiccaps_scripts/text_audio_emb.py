import os
import pickle

import datasets
import soundfile as sf
from omegaconf import DictConfig

from mmda.utils.audio_utils import convert_to_mono_channel, resample_audio
from mmda.utils.get_embeddings import clap_audio, clap_text, clip_text, gtr_text
from mmda.utils.hydra_utils import hydra_main

BATCH_SIZE = 32

@hydra_main(version_base=None, config_path='../config', config_name='musiccaps')
def MusicCaps_embed_audio_text(cfg: DictConfig):
    dataset = datasets.load_dataset('google/MusicCaps', split='train')
    audio_list, caption_list = [], []
    for data in dataset:
        audio_path = os.path.join(cfg.paths.dataset_path, f"{data['ytid']}.wav")
        caption = data["caption"]
        # print(f"Processing audio file {audio_path} with caption: {caption}")
        ### check if the audio file exists
        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} does not exist. Skipping.")
            continue
        audio, sample_rate = sf.read(audio_path)
        audio = convert_to_mono_channel(audio, normalize=True)
        audio = resample_audio(audio, sample_rate, 48000)

        audio_list.append(audio)
        caption_list.append(caption)
        # if len(audio_list) == 100:
        #     break

    print(f"Number of audio files: {len(audio_list)}. Number of captions: {len(caption_list)}")

    if not os.path.exists(cfg.paths.save_path):
        os.makedirs(cfg.paths.save_path)

    clap_text_features = clap_text(caption_list, batch_size=BATCH_SIZE)
    print(clap_text_features.shape)
    with open(cfg.paths.save_path + 'MusicCaps_text_emb_clap.pkl', 'wb') as f:
        pickle.dump(clap_text_features, f)
    
    clip_text_features = clip_text(caption_list, batch_size=BATCH_SIZE)
    print(clip_text_features.shape)
    with open(cfg.paths.save_path + 'MusicCaps_text_emb_clip.pkl', 'wb') as f:
        pickle.dump(clip_text_features, f)

    gtr_text_features = gtr_text(caption_list)
    print(gtr_text_features.shape)
    with open(cfg.paths.save_path + 'MusicCaps_text_emb_gtr.pkl', 'wb') as f:
        pickle.dump(gtr_text_features, f)

    clap_audio_features = clap_audio(audio_list, batch_size=BATCH_SIZE)
    print(clap_audio_features.shape)
    with open(cfg.paths.save_path + 'MusicCaps_audio_emb_clap.pkl', 'wb') as f:
        pickle.dump(clap_audio_features, f)

    return


if __name__ == "__main__": 
    MusicCaps_embed_audio_text()

# CUDA_VISIBLE_DEVICES=4 poetry run python text_audio_emb.py 