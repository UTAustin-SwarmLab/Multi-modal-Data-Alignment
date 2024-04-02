import pickle

from omegaconf import DictConfig

from mmda.utils.data_utils import (
    load_SOP,
)
from mmda.utils.get_embeddings import clip_text, gtr_text
from mmda.utils.hydra_utils import hydra_main

BATCH_SIZE = 64


@hydra_main(version_base=None, config_path="../config", config_name="sop")
def main(cfg: DictConfig):  # noqa: D103
    _, text_descriptions = load_SOP()

    # get text embeddings
    print("Loading CLIP")
    text_emb = clip_text(text_descriptions, BATCH_SIZE)  # batched np array
    print("CLIP embeddings done:", text_emb.shape)
    with open(cfg.paths.save_path + "data/SOP_text_emb_clip.pkl", "wb") as f:
        pickle.dump(text_emb, f)

    print("Loading GTR")
    text_emb = gtr_text(text_descriptions, BATCH_SIZE)  # batched np array
    print("GTR embeddings done:", text_emb.shape)
    with open(cfg.paths.save_path + "data/SOP_text_emb_gtr.pkl", "wb") as f:
        pickle.dump(text_emb, f)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=4 python SOP_text_emb.py
