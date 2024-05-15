"""This script is used to detect mislabeled data in the bimodal datasets."""

from omegaconf import DictConfig

import hydra
from mmda.exps.linear_encoder import linear_exps


@hydra.main(version_base=None, config_path="../config", config_name="linear")
def main(cfg: DictConfig) -> None:
    """Main function for the linear encoder experiment.

    Args:
        cfg: config file
    """
    linear_exps(cfg)


if __name__ == "__main__":
    main()
