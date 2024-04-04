import matplotlib.pyplot as plt
from omegaconf import DictConfig
from swarm_visualizer.scatterplot import plot_basic_scatterplot
from swarm_visualizer.utility.general_utils import save_fig

from mmda.spearman_coeff import cal_spearman_coeff
from mmda.utils.data_utils import load_dataset_config
from mmda.utils.hydra_utils import hydra_main


@hydra_main(version_base=None, config_path="../config", config_name="main")
def Spearman_coeff(cfg: DictConfig):
    """Calculate the Spearman's rank coeff of MusicCaps with CLAP model and CCA similarity score.

    Args:
        cfg: Config dictionary.
    """
    cfg_dataset = load_dataset_config(cfg)
    r, p_value, sim_score_CLAP, sim_score_CCA, rank_CLAP, rank_CCA = cal_spearman_coeff(cfg)

    print(f"Spearman's rank coefficient: {r}")
    print(f"p-value: {p_value}")

    # plot scatter plot of similarity score's ranks
    fig_rank, ax_rank = plt.subplots()
    plot_basic_scatterplot(
        ax=ax_rank,
        x=rank_CLAP,
        y=rank_CCA,
        xlabel="CLAP similarity score rank",
        ylabel="CCA similarity score rank",
    )
    save_fig(fig_rank, cfg_dataset.paths.plots_path + "sim_rank_scatter_plot.png", dpi=400)

    # plot scatter plot of the original similarity scores
    fig, ax = plt.subplots()
    plot_basic_scatterplot(
        ax=ax,
        x=sim_score_CLAP,
        y=sim_score_CCA,
        xlabel="CLAP similarity score",
        ylabel="CCA similarity score",
    )
    save_fig(fig, cfg_dataset.paths.plots_path + "sim_scatter_plot.png", dpi=400)
    return


if __name__ == "__main__":
    Spearman_coeff()
