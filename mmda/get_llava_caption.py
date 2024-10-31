"""Get image captions from llava model and save them."""

import pickle
from pathlib import Path

import joblib
from omegaconf import DictConfig

import hydra
from mmda.liploc.dataloaders.KittiBothDataset import KITTIBothDataset
from mmda.utils.liploc_model import CFG, load_eval_filenames
from mmda.utils.llava_utils import llava_caption


@hydra.main(version_base=None, config_path="../config", config_name="main")
def get_caption(cfg: DictConfig) -> None:
    """Get captions from llava model and save them.

    Args:
        cfg: config
    """
    cfg_dataset = cfg[cfg.dataset]
    if cfg.dataset == "sop":
        path_text_descriptions = joblib.load(
            Path(cfg_dataset.paths.dataset_path + "text_descriptions_SOP.pkl")
        )
        for path_text in path_text_descriptions:
            path_text[0] = path_text[0].replace("/store/", "/nas/")
        img_paths = [path_text[0] for path_text in path_text_descriptions]
    elif cfg.dataset == "pitts":
        # val set
        val_path_text_descriptions = joblib.load(
            "/nas/omama/datasets/pitts250k/text_descriptions_pitts30k.pkl"
        )
        for path_text in val_path_text_descriptions:
            path_text[0] = (
                path_text[0]
                .replace("/store/", "/nas/")
                .replace("TextMapReduce", "datasets")
            )
        path_text_descriptions = val_path_text_descriptions
        img_paths = [path_text[0] for path_text in path_text_descriptions]
    elif cfg.dataset == "KITTI":
        filenames = load_eval_filenames()
        dataset = KITTIBothDataset(
            transforms=[],
            CFG=CFG,
            filenames=filenames,
        )
        img_paths, lidar_paths = dataset.get_image_lidar_paths()
    # TODO: add more datasets
    else:
        msg = f"Dataset {cfg.dataset} not supported."
        raise ValueError(msg)

    # query llava without shuffling
    llava_captions = llava_caption(cfg, img_paths)
    model_name = cfg.llava.model_path.split("/")[-1]

    Path(cfg_dataset.paths.dataset_path).mkdir(parents=True, exist_ok=True)
    # Save text_descriptions joblib
    with Path(
        cfg_dataset.paths.dataset_path + f"{cfg.dataset}_{model_name}_captions.pkl"
    ).open("wb") as f:
        pickle.dump(llava_captions, f)


if __name__ == "__main__":
    get_caption()
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 poetry run python mmda/get_llava_caption.py
