"""This script is used to retrieve multimodal datasets."""

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

import hydra
from mmda.exps.any2any_retrieval import any2any_retrieval


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main function to generate the retrieval results of the multimodal datasets.

    Args:
        cfg: config file
    """
    assert (
        cfg.dataset in cfg.any_retrieval_datasets
    ), f"{cfg.dataset} is not for any2any retrieval."
    (maps, precisions, recalls), (full_maps, full_precisions, full_recalls) = (
        any2any_retrieval(cfg)
    )

    print(f"mAP@5: {maps[5]}, mAP@20: {maps[20]}")
    print(
        f"Precision@1: {precisions[1]}, Precision@5: {precisions[5]}, Precision@20: {precisions[20]}"
    )
    print(f"Recall@1: {recalls[1]}, Recall@5: {recalls[5]}, Recall@20: {recalls[20]}")
    print(f"mAP@5: {full_maps[5]}, mAP@20: {full_maps[20]}")
    print(
        f"Precision@1: {full_precisions[1]}, Precision@5: {full_precisions[5]}, Precision@20: {full_precisions[20]}"
    )
    print(
        f"Recall@1: {full_recalls[1]}, Recall@5: {full_recalls[5]}, Recall@20: {full_recalls[20]}"
    )

    # write the results to a csv file
    cfg_dataset = cfg[cfg.dataset]
    data = {
        "method": [
            "Conformal Retrieval (Missing Data)",
            "Conformal Retrieval (Full Data)",
        ],
        "mAP@5": [maps[5], full_maps[5]],
        "mAP@20": [maps[20], full_maps[20]],
        "Precision@1": [precisions[1], full_precisions[1]],
        "Precision@5": [precisions[5], full_precisions[5]],
        "Precision@20": [precisions[20], full_precisions[20]],
        "Recall@1": [recalls[1], full_recalls[1]],
        "Recall@5": [recalls[5], full_recalls[5]],
        "Recall@20": [recalls[20], full_recalls[20]],
    }
    df = pd.DataFrame(data)
    df_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"any2any_retrieval_{cfg_dataset.sim_dim}_{cfg_dataset.mask_ratio}.csv"
    )
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(df_path, index=False)


if __name__ == "__main__":
    main()
# poetry run python mmda/any2any_conformal_retrieval.py
