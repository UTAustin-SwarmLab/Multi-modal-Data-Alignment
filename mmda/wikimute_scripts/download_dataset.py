import os
import time

import datasets
import soundfile as sf
import wget
from omegaconf import DictConfig

from mmda.utils.audio_utils import convert_to_mono_channel, resample_audio
from mmda.utils.hydra_utils import hydra_main


@hydra_main(version_base=None, config_path='../config', config_name='wikimute')
def download_dataset(cfg: DictConfig):
    dataset = datasets.load_dataset("davanstrien/WikiMuTe")
    dataset.save_to_disk(cfg.paths.dataset_path)
    dataset = datasets.load_from_disk(cfg.paths.dataset_path)
    for data in dataset['train']:
        audio_url = data["audio_url"] # https://upload.wikimedia.org/wikipedia/en/6/61/"24K_Magic".ogg
        sentence = data["sentences"] # list of sentences
        song_name = data["file"] # "24K Magic".ogg
        aspects = data["aspects"]
        pageid = data["pageid"]

        if not os.path.exists(os.path.join(cfg.paths.dataset_path, "ogg_files")):
            os.makedirs(os.path.join(cfg.paths.dataset_path, "ogg_files"))
        ogg_save_path = os.path.join(cfg.paths.dataset_path, "ogg_files", song_name)
        if not os.path.exists(ogg_save_path):
            time.sleep(1)
            audio_file = wget.download(audio_url, out=ogg_save_path)

    return

    audio, sample_rate = sf.read(audio_file)
    # convert to mono channel
    audio = convert_to_mono_channel(audio, normalize=True)
    # resample to 48kHz
    audio = resample_audio(audio, orig_sr=sample_rate, target_sr=48000)


if __name__ == "__main__": 
    download_dataset()
