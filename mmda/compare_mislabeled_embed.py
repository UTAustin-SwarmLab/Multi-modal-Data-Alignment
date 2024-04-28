"""This script is used to compare the correct and mislabeled embeddings in the datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from swarm_visualizer.histogram import plot_several_pdf
from swarm_visualizer.utility.general_utils import save_fig

import hydra
from mmda.exps.mislabel_align import parse_wrong_label
from mmda.utils.data_utils import load_two_encoder_data
from mmda.utils.sim_utils import cosine_sim


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: D103

    cfg_dataset, data1, data2 = load_two_encoder_data(cfg)
    assert cfg.dataset == "cosmos", "Only cosmos dataset is supported."

    plots_path = Path(cfg_dataset.paths.plots_path)

    plots_path.mkdir(parents=True, exist_ok=True)
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

    mislabel_cos_sim = cosine_sim(data2_align, data2_unalign)
    print(
        "mean cos_sim", mislabel_cos_sim.mean(), "std cos_sim", mislabel_cos_sim.std()
    )

    # similarity score of random pairs of data
    np.random.seed(cfg.seed)
    data2_new = data2.copy()
    np.random.shuffle(data2_new)
    shuffle_cos_sim = cosine_sim(data2, data2_new)
    print("mean cos_sim", shuffle_cos_sim.mean(), "std cos_sim", shuffle_cos_sim.std())

    # sanity check
    img_cos_sim = cosine_sim(data1[idx_corrct], data1[idx_wrong])
    assert np.allclose(img_cos_sim, 1.0, atol=1e-4)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(
        data_list=[mislabel_cos_sim, shuffle_cos_sim],
        legend=["Misaligned", "Random"],
        title_str="Similarity Score Distribution",
        xlabel="Cosine similarity Score",
        ylabel="Frequency",
        ax=ax,
        binwidth=0.05,
    )
    save_fig(
        fig,
        plots_path / "cos_similarity_compare.png",
    )


if __name__ == "__main__":
    main()
