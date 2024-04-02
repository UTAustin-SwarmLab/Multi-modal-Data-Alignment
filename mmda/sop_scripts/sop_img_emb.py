import pickle

from omegaconf import DictConfig

from mmda.utils.data_utils import (
    load_SOP,
)
from mmda.utils.get_embeddings import clip_imgs, dinov2
from mmda.utils.hydra_utils import hydra_main

BATCH_SIZE = 64


@hydra_main(version_base=None, config_path="../config", config_name="sop")
def main(cfg: DictConfig):  # noqa: D103
    img_files, _ = load_SOP(cfg)

    # get img embeddings
    img_emb = dinov2(img_files, BATCH_SIZE)  # batched np array
    with open(cfg.paths.save_path + "data/SOP_img_emb_dino.pkl", "wb") as f:
        pickle.dump(img_emb, f)
    print("DINO embeddings saved")

    img_emb = clip_imgs(img_files, BATCH_SIZE)  # batched np array
    with open(cfg.paths.save_path + "data/SOP_img_emb_clip.pkl", "wb") as f:
        pickle.dump(img_emb, f)
    print("CLIP embeddings saved")

    # print("Loading ViT")
    # img_emb = vit(train_test_img_files, BATCH_SIZE) # batched np array
    # with open(save_dir + 'data/SOP_img_emb_train_test_vit.pkl', 'wb') as f:
    #     pickle.dump(img_emb, f)
    # print("ViT embeddings saved")


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 python SOP_img_emb.py
