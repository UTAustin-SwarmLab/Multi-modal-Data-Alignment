"""This script is used to detect mislabeled data in the bimodal datasets."""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from mmda.exps.multimodal_retrieval import (
    cca_retrieval,
    clip_like_retrieval,
)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: D103
    cfg_dataset = cfg[cfg.dataset]

    clip_recall, clip_precision = clip_like_retrieval(cfg)
    print(f"CLIP-like: Recall: {clip_recall}, Precision: {clip_precision}")

    cca_recalls, cca_precisions = cca_retrieval(cfg)
    print(f"CCA: Recall: {cca_recalls}, Precision: {cca_precisions}")

    # write to csv file
    data = {
        "method": ["CLIP-like"] + ["CCA" for _ in cca_recalls],
        "projection dim": [1280, *cca_recalls.keys()],
        "recall@1": [clip_recall[1]]
        + [cca_recall[1] for cca_recall in cca_recalls.values()],
        "recall@5": [clip_recall[5]]
        + [cca_recall[5] for cca_recall in cca_recalls.values()],
        "precision@1": [clip_precision[1]]
        + [cca_precision[1] for cca_precision in cca_precisions.values()],
        "precision@5": [clip_precision[5]]
        + [cca_precision[5] for cca_precision in cca_precisions.values()],
    }
    df = pd.DataFrame(data)
    img2text_label = "img2text" if cfg_dataset.img2text else "text2img"
    sim_dim = cfg_dataset.sim_dim
    df_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"retrieval_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}"
        / f"{img2text_label}_results_{sim_dim}.csv"
    )
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(df_path, index=False)


if __name__ == "__main__":
    main()
