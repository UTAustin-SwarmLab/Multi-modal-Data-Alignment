
import datasets
from omegaconf import DictConfig

from mmda.utils.hydra_utils import hydra_main


@hydra_main(version_base=None, config_path='../config', config_name='audiocaps')
def download_dataset(cfg: DictConfig):
    ds = datasets.load_dataset("jp1924/AudioCaps") 
    ds.save_to_disk(cfg.paths.dataset_path)
    ds = datasets.load_from_disk(cfg.paths.dataset_path)

    # datasets.config.DOWNLOADED_DATASETS_PATH = Path(cfg.paths.dataset_path)
    # _ = datasets.load_dataset("jp1924/AudioCaps") 

if __name__ == "__main__": 
    download_dataset()
