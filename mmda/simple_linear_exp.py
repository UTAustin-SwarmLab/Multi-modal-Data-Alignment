"""This script is used to detect mislabeled data in the bimodal datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig
from swarm_visualizer import plot_overlaid_lineplot, plot_paired_boxplot
from swarm_visualizer.utility import add_wilcoxon_value, save_fig, set_plot_properties

import hydra
from mmda.exps.linear_encoder import linear_exps


@hydra.main(version_base=None, config_path="../config", config_name="linear")
def main(cfg: DictConfig) -> None:
    """Main function for the linear encoder experiment.

    Args:
        cfg: config file
    """
    snr_list, lambda_list, sim_score_list, sim_score_shuffled_list = linear_exps(cfg)

    # plot the ROC curve
    fig, ax = plt.subplots()
    plots_path = Path(cfg.paths.plots_path)
    plots_path.mkdir(parents=True, exist_ok=True)
    eq_label = "_noweight" if cfg.equal_weights else ""

    set_plot_properties(autolayout=True)
    s_range = range(1, len(sim_score_list) + 1)
    data_dict = {
        "SNR (\u2191 higher is better)": {
            "x": s_range,
            "y": snr_list,
            "lw": 2,
            "linestyle": "-",
            "color": "blue",
        },
        "Lambda (\u2191 higher is better)": {
            "x": s_range,
            "y": lambda_list,
            "lw": 2,
            "linestyle": "-.",
            "color": "red",
        },
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_overlaid_lineplot(
        ax=ax,
        normalized_dict=data_dict,
        title_str="SNR and Lambda",
        ylabel="dB",
        xlabel="Selected dimension for Similarity score score $s$",
        legend_present=True,
    )
    save_fig(
        fig,
        plots_path / f"snr_lambda{eq_label}.png",
        dpi=600,
        tight_layout=True,
    )

    # create a dataframe
    set_plot_properties(autolayout=True)
    df = pd.DataFrame(
        columns=[
            "Selected dimension for Similarity score score $s$",
            "Similarity score",
            "shuffled",
        ]
    )
    boxpairs = []
    for s, (sim_scores, sim_score_shuffled) in enumerate(
        zip(sim_score_list, sim_score_shuffled_list, strict=True)
    ):
        if s % 5 != 0:
            continue
        boxpairs.append(((s + 1, "original"), (s + 1, "shuffled")))
        for score in sim_scores:
            df.loc[len(df.index)] = [s + 1, score, "original"]
        for score in sim_score_shuffled:
            df.loc[len(df.index)] = [s + 1, score, "shuffled"]
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_paired_boxplot(
        ax,
        df,
        x_var="Selected dimension for Similarity score score $s$",
        y_var="Similarity score",
        hue="shuffled",
        showfliers=False,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right")
    # calculate the p-value
    add_wilcoxon_value(
        ax=ax,
        df=df,
        x_var="Selected dimension for Similarity score score $s$",
        y_var="Similarity score",
        hue="shuffled",
        box_pairs=boxpairs,
        test_type="Wilcoxon",
        text_format="full",
        loc="inside",
        fontsize=14,
        verbose=2,
        pvalue_format_string="{:.4f}",
        show_test_name=False,
    )
    save_fig(
        fig, plots_path / f"similarity_score{eq_label}.png", dpi=600, tight_layout=True
    )


if __name__ == "__main__":
    main()
