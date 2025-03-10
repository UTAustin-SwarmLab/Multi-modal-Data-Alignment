"""Stanford_Online_Products specific. User can input the image name to query the text description."""

import numpy as np
from omegaconf import DictConfig

import hydra
from mmda.utils.dataset_utils import load_sop


@hydra.main(version_base=None, config_path="../config", config_name="main")
def sop_print_text(cfg: DictConfig) -> None:  # noqa: D103
    while True:
        # load raw data
        img_paths, text_descriptions, classes, obj_ids = load_sop(cfg.sop)
        img_names = [
            img_path.split("/")[-1].replace(".JPG", "") for img_path in img_paths
        ]
        img_text_dict = dict(zip(img_names, text_descriptions, strict=True))
        print(np.unique(classes))

        # print an image's text description of bicycle and cabinet
        bicycle_idx = np.where(np.array(classes) == "bicycle")[0]
        print("Bicycle example:", len(bicycle_idx))
        for i in range(10):
            print(img_names[bicycle_idx[i]], img_text_dict[img_names[bicycle_idx[i]]])

        cabinet_idx = np.where(np.array(classes) == "cabinet")[0]
        print("Cabinet example:", len(cabinet_idx))
        for i in range(10):
            print(img_names[cabinet_idx[i]], img_text_dict[img_names[cabinet_idx[i]]])

        img_name = input("Query image name: ")
        if img_name not in img_text_dict:
            print("Image name not found.")
        else:
            print(img_text_dict[img_name])


if __name__ == "__main__":
    sop_print_text()
