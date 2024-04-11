import os
import pickle

from omegaconf import DictConfig

import hydra
from mmda.utils.data_utils import load_dataset_config
from mmda.utils.llava_utils import llava_caption


@hydra.main(version_base=None, config_path="../config", config_name="main")
def get_caption(cfg: DictConfig):
    """Get captions from llava model and save them.

    Args:
        cfg: config
    """
    cfg_dataset = load_dataset_config(cfg)
    if cfg.dataset == "sop":
        with open(cfg_dataset.paths.dataset_path + "text_descriptions_SOP.pkl", "rb") as f:
            path_text_descriptions = pickle.load(f)
    elif cfg.dataset == "pitts":
        # # train set
        # with open(cfg_dataset.paths.dataset_path + "text_descriptions_pitts30k_train.pkl", "rb") as f:
        #     train_path_text_descriptions = pickle.load(f)
        # val set
        with open(cfg_dataset.paths.dataset_path + "text_descriptions_pitts30k.pkl", "rb") as f:
            val_path_text_descriptions = pickle.load(f)
        path_text_descriptions = val_path_text_descriptions
        print("Number of images in the dataset:", len(path_text_descriptions))
    # TODO: add more datasets
    else:
        raise ValueError(f"Dataset {cfg.dataset} not supported.")

    for path_text in path_text_descriptions:
        # /store/omama/datasets/sop/000/000426_pitch1_yaw1.jpg
        path_text[0] = path_text[0].replace("/store/", "/nas/")
    img_paths = [path_text[0] for path_text in path_text_descriptions]

    # query llava without shuffling
    llava_captions = llava_caption(cfg, img_paths)
    model_name = cfg.llava.model_path.split("/")[-1]

    os.makedirs(cfg_dataset.paths.dataset_path, exist_ok=True)
    # Save text_descriptions pickle
    with open(
        cfg_dataset.paths.dataset_path + f"{cfg.dataset}_{model_name}_captions.pkl",
        "wb",
    ) as f:
        pickle.dump(llava_captions, f)


if __name__ == "__main__":
    get_caption()
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 poetry run python scripts/get_llava_caption.py
