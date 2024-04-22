"""Get image captions from llava model and save them."""

import pickle
from pathlib import Path

import joblib
from omegaconf import DictConfig

import hydra
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
    elif cfg.dataset == "pitts":
        # val set
        val_path_text_descriptions = joblib.load(
            "/nas/omama/datasets/pitts250k/text_descriptions_pitts30k.pkl"
        )
        # /store/omama/TextMapReduce/pitts250k/000/000000_pitch1_yaw1.jpg
        for path_text in val_path_text_descriptions:
            path_text[0] = (
                path_text[0]
                .replace("/store/", "/nas/")
                .replace("TextMapReduce", "datasets")
            )
        path_text_descriptions = val_path_text_descriptions
    # TODO: add more datasets
    else:
        msg = f"Dataset {cfg.dataset} not supported."
        raise ValueError(msg)

    img_paths = [path_text[0] for path_text in path_text_descriptions]
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
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 poetry run python mmda/get_llava_caption.py
