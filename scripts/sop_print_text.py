"""Stanford_Online_Products specific. User can input the image name to query the text description."""

import numpy as np
from omegaconf import DictConfig

from mmda.utils.data_utils import (
    load_SOP,
)
from mmda.utils.hydra_utils import hydra_main


@hydra_main(version_base=None, config_path="../config", config_name="main")
def SOP_print_text(cfg: DictConfig):  # noqa: D103
    while True:
        # load raw data
        img_paths, text_descriptions, classes, obj_ids = load_SOP(cfg.sop)
        img_names = [img_path.split("/")[-1].replace(".JPG", "") for img_path in img_paths]
        img_text_dict = {img_name: text for img_name, text in zip(img_names, text_descriptions)}
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
    SOP_print_text()
