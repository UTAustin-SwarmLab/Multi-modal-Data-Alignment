"""This script is used to retrieve multimodal datasets."""

from omegaconf import DictConfig

import hydra
from mmda.exps.any_retrieval import any2any_retrieval


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main function to generate the retrieval results of the multimodal datasets.

    Args:
        cfg: config file
    """
    assert (
        cfg.dataset in cfg.any_retrieval_datasets
    ), f"{cfg.dataset} is not for any2any retrieval."
    any2any_retrieval(cfg)


if __name__ == "__main__":
    main()
