"""This script is used to detect mislabeled data in the bimodal datasets."""

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

import hydra
from mmda.exps.retrieval import (
    cca_retrieval,
    clip_like_retrieval,
)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: D103
    cfg_dataset = cfg[cfg.dataset]

    clip_maps, clip_precision = clip_like_retrieval(cfg)
    print(f"CLIP-like: mAP: {clip_maps}, Precision: {clip_precision}")

    cca_maps_dict, cca_precisions = cca_retrieval(cfg)
    print(f"CCA: mAP: {cca_maps_dict}, Precision: {cca_precisions}")

    # write to csv file
    data = {
        "method": ["CLIP-like"] + ["CCA" for _ in cca_maps_dict],
        "projection dim": [1280, *cca_maps_dict.keys()],
        "mAP": [clip_maps, *list(cca_maps_dict.values())],
        "precision@1": [clip_precision[1]]
        + [cca_precision[1] for cca_precision in cca_precisions.values()],
        "precision@5": [clip_precision[5]]
        + [cca_precision[5] for cca_precision in cca_precisions.values()],
    }
    df = pd.DataFrame(data)
    img2text_label = "img2text" if cfg_dataset.img2text else "text2img"
    sim_dim = cfg_dataset.sim_dim
    eq_label = "_noweight" if cfg_dataset.equal_weights else ""
    df_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"retrieval_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}"
        / f"{img2text_label}_results_{sim_dim}_{eq_label}.csv"
    )
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(df_path, index=False)


if __name__ == "__main__":
    main()
