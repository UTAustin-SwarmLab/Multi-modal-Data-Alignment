### Download the clips within the MusicCaps dataset from YouTube. 
### modified from https://colab.research.google.com/github/nateraw/download-musiccaps-dataset/blob/main/download_musiccaps.ipynb#scrollTo=FV-nFNShP7Xd
import os
import subprocess
from pathlib import Path

from datasets import Audio, load_dataset
from omegaconf import DictConfig

from mmda.utils.hydra_utils import hydra_main


def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir='/tmp/musiccaps',
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    status = False

    command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" {url_base}{video_identifier}
    """.strip()

    attempts = 0
    while True:
        try:
            _ = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def load_dataset_and_download(
    data_dir: str,
    sampling_rate: int = 44100,
    limit: int = None,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
):
    """
    Download the clips within the MusicCaps dataset from YouTube.
    Args:
        data_dir: Directory to save the clips to.
        sampling_rate: Sampling rate of the audio clips.
        limit: Limit the number of examples to download.
        num_proc: Number of processes to use for downloading.
        writer_batch_size: Batch size for writing the dataset. This is per process.
    """

    ds = load_dataset('google/MusicCaps', split='train')
    if limit is not None:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            status = False
            status, log = download_clip(
                example['ytid'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )

        example['audio'] = outfile_path
        example['download_status'] = status
        return example

    return ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    ).cast_column('audio', Audio(sampling_rate=sampling_rate))


@hydra_main(version_base=None, config_path='../config', config_name='musiccaps')
def main(cfg: DictConfig):
    load_dataset_and_download(cfg.paths.dataset_path, num_proc=10)
    
    # if not os.path.exists(cfg.paths.dataset_path):
    #     os.makedirs(cfg.paths.dataset_path)
    # print(f"Saving dataset to {cfg.paths.dataset_path}")
    # ds.save_to_disk(cfg.paths.dataset_path)

if __name__ == '__main__':
    main()