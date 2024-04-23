"""This script is used to detect mislabeled data in the bimodal datasets."""

from omegaconf import DictConfig

import hydra
from mmda.exps.multimodal_retrieval import (
    asif_retrieval,
    cca_retrieval,
    clip_like_retrieval,
)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: D103
    clip_model_name = "CLAP" if cfg.dataset == "musiccaps" else "CLIP"

    recall, precision = cca_retrieval(cfg)
    print(f"CCA: Recall: {recall}, Precision: {precision}")

    recall, precision = asif_retrieval(cfg)
    print(f"ASIF: Recall: {recall}, Precision: {precision}")

    recall, precision = clip_like_retrieval(cfg, clip_model_name)
    print(f"CLIP-like: Recall: {recall}, Precision: {precision}")


if __name__ == "__main__":
    main()
