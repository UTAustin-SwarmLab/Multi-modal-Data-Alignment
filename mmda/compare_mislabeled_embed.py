"""This script is used to compare the correct and mislabeled embeddings in the datasets."""

import numpy as np
from omegaconf import DictConfig

import hydra
from mmda.exps.data_mislabel_align import (
    parse_wrong_label,
)
from mmda.utils.data_utils import load_two_encoder_data
from mmda.utils.sim_utils import cosine_sim


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: D103

    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)

    assert cfg.dataset == "cosmos", "Only cosmos dataset is supported."

    # similarity score of correct and mislabeled data
    wrong_labels_bool = parse_wrong_label(cfg)
    idx_wrong = np.array([i for i, x in enumerate(wrong_labels_bool) if x])
    idx_corrct = idx_wrong - 1
    print("number of mislabeled data", len(idx_wrong))
    print("number of correct data", len(idx_corrct))
    data2_align = data2[idx_corrct]
    data2_unalign = data2[idx_wrong]
    print("data2_align shape", data2_align.shape)
    print("data2_unalign shape", data2_unalign.shape)

    cos_sim = cosine_sim(data2_align, data2_unalign)
    print("mean cos_sim", cos_sim.mean(), "std cos_sim", cos_sim.std())

    # similarity score of random pairs of data
    np.random.seed(cfg.seed)
    data2_new = data2.copy()
    np.random.shuffle(data2_new)
    cos_sim = cosine_sim(data2, data2_new)
    print("mean cos_sim", cos_sim.mean(), "std cos_sim", cos_sim.std())

    # sanity check
    cos_sim = cosine_sim(data1[idx_corrct], data1[idx_wrong])
    assert np.allclose(cos_sim, 1.0, atol=1e-4)


if __name__ == "__main__":
    main()
