"""calculate the spearman's rank coefficient with CLIP model's and our proposed method's similarity score."""

from omegaconf import DictConfig

import hydra
from mmda.exps.spearman_coeff import retrieval_spearman_coeff


@hydra.main(version_base=None, config_path="../config", config_name="main")
def spearman_coeff_cross_retrieval(cfg: DictConfig) -> None:
    """Calculate the Spearman's rank coeff of CLIP model and CCA similarity score for retrieval tasks.

    Args:
        cfg: Config dictionary.
    """
    avg_r, avg_p_value = retrieval_spearman_coeff(cfg)

    print(f"Spearman's rank coefficient: {avg_r}")
    print(f"p-value: {avg_p_value}")


if __name__ == "__main__":
    spearman_coeff_cross_retrieval()
