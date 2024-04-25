"""calculate the spearman's rank coefficient with CLIP model's and our proposed method's similarity score."""

from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig
from swarm_visualizer.scatterplot import plot_basic_scatterplot
from swarm_visualizer.utility.general_utils import save_fig

import hydra
from mmda.exps.spearman_coeff import cal_spearman_coeff


@hydra.main(version_base=None, config_path="../config", config_name="main")
def spearman_coeff(cfg: DictConfig) -> None:
    """Calculate the Spearman's rank coeff of MusicCaps with CLAP model and CCA similarity score.

    Args:
        cfg: Config dictionary.
    """
    cfg_dataset = cfg[cfg.dataset]
    r, p_value, sim_score_clip, sim_score_cca, rank_clap, rank_cca = cal_spearman_coeff(
        cfg
    )

    print(f"Spearman's rank coefficient: {r}")
    print(f"p-value: {p_value}")

    # plot scatter plot of similarity score's ranks
    fig_rank, ax_rank = plt.subplots()
    plot_basic_scatterplot(
        ax=ax_rank,
        x=rank_clap,
        y=rank_cca,
        xlabel="CLAP similarity score rank",
        ylabel="CCA similarity score rank",
    )
    plot_path = Path(cfg_dataset.paths.plots_path)
    plot_path.mkdir(parents=True, exist_ok=True)
    save_fig(fig_rank, plot_path / "sim_rank_scatter_plot.png", dpi=400)

    # plot scatter plot of the original similarity scores
    fig, ax = plt.subplots()
    plot_basic_scatterplot(
        ax=ax,
        x=sim_score_clip,
        y=sim_score_cca,
        xlabel="CLIP similarity score",
        ylabel="CCA similarity score",
    )
    save_fig(fig, plot_path / "sim_scatter_plot.png", dpi=400)


if __name__ == "__main__":
    spearman_coeff()
