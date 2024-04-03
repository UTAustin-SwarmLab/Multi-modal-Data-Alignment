import matplotlib.pyplot as plt
import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig
from swarm_visualizer.histogram import (
    plot_several_pdf,
)
from swarm_visualizer.utility.general_utils import save_fig

from mmda.utils.data_utils import (
    filter_str_label,
    get_train_test_split_index,
    load_CLIP_like_data,
    load_SOP,
    load_two_encoder_data,
    origin_centered,
    shuffle_data_by_indices,
    train_test_split,
)
from mmda.utils.sim_utils import (
    ROC_points,
    cosine_sim,
    weighted_corr_sim,
)


def CCA_data_align(cfg: DictConfig, shuffle_level: str = "dataset") -> list[tuple[float, float]]:
    """Align the audio and text data and calculate the similarity score using my proposed method.

    Args:
        cfg: configuration file
        shuffle_level: shuffle level. It can be "dataset", "class", or "object".

    Returns:
        ROC points

    """
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, Data1, Data2 = load_two_encoder_data(cfg)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Data1.shape[0])
    trainData1, valData1 = train_test_split(Data1, trainIdx, valIdx)
    trainData2, valData2 = train_test_split(Data2, trainIdx, valIdx)

    ### aligned case: not shuffle the data
    trainData1Align, valData1Align = trainData1.copy(), valData1.copy()
    trainData2Align, valData2Align = trainData2.copy(), valData2.copy()
    # zero mean data
    trainData1Align, trainData1_mean = origin_centered(trainData1Align)
    trainData2Align, trainData2_mean = origin_centered(trainData2Align)
    valData1Align = valData1Align - trainData1_mean
    valData2Align = valData2Align - trainData2_mean
    # make sure the data is zero mean
    assert np.allclose(
        trainData1Align.mean(axis=0), 0, atol=1e-4
    ), f"trainData1Align not zero mean: {trainData1Align.mean(axis=0)}"
    assert np.allclose(
        trainData2Align.mean(axis=0), 0, atol=1e-4
    ), f"trainData2Align not zero mean: {trainData2Align.mean(axis=0)}"

    # CCA dimensionality reduction
    cca = CCA(latent_dimensions=cfg_dataset.CCA_dim)
    trainData1Align, trainData2Align = cca.fit_transform((trainData1Align, trainData2Align))
    if cfg_dataset.equal_weights:
        corr_align = np.ones((trainData2Align.shape[1],))  # dim,
    else:
        corr_align = np.diag(trainData1Align.T @ trainData2Align) / trainData1Align.shape[0]  # dim,

    # calculate the similarity score
    valData1Align, valData2Align = cca.transform((valData1Align, valData2Align))
    sim_align = weighted_corr_sim(valData1Align, valData2Align, corr_align, dim=cfg_dataset.sim_dim)

    ### unaligned case: shuffle the data
    # shuffle only the text data
    trainData1Unalign, valData1Unalign = trainData1.copy(), valData1.copy()
    trainData2Unalign, valData2Unalign = trainData2.copy(), valData2.copy()

    valData2Unalign = shuffle_by_level(cfg_dataset, cfg.dataset, shuffle_level, valData2Unalign, trainIdx, valIdx)

    # zero mean data
    trainData1Unalign, trainData1_mean_ = origin_centered(trainData1Unalign)
    trainData2Unalign, trainData2_mean_ = origin_centered(trainData2Unalign)
    valData1Unalign = valData1Unalign - trainData1_mean_
    valData2Unalign = valData2Unalign - trainData2_mean_

    # make sure the data is zero mean
    assert np.allclose(
        trainData1Unalign.mean(axis=0), 0, atol=1e-4
    ), f"trainData1Unalign not zero mean: {trainData1Unalign.mean(axis=0)}"
    assert np.allclose(
        trainData2Unalign.mean(axis=0), 0, atol=1e-4
    ), f"trainData2Unalign not zero mean: {trainData2Unalign.mean(axis=0)}"

    valData1Align, valData2Align = cca.transform((valData1Unalign, valData2Unalign))
    sim_unalign = weighted_corr_sim(valData1Align, valData2Align, corr_align, dim=cfg_dataset.sim_dim)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(
        data_list=[sim_align, sim_unalign],
        legend=["Aligned", "Unaligned"],
        title_str="Similarity Score Distribution",
        xlabel="Similarity Score",
        ylabel="Frequency",
        ax=ax,
    )
    if cfg_dataset.equal_weights:
        save_fig(
            fig,
            cfg_dataset.paths.plots_path
            + f"similarity_score_{shuffle_level}_r{cfg.train_test_ratio}_dim{cfg_dataset.sim_dim}_noweight.png",
        )
    else:
        save_fig(
            fig,
            cfg_dataset.paths.plots_path
            + f"similarity_score_{shuffle_level}_r{cfg.train_test_ratio}_dim{cfg_dataset.sim_dim}.png",
        )

    # CCA dimensionality reduction
    cca_unalign = CCA(latent_dimensions=cfg_dataset.CCA_dim)
    trainData1Unalign, trainData2Unalign = cca_unalign.fit_transform((trainData1Unalign, trainData2Unalign))
    corr_unalign = np.diag(trainData1Unalign.T @ trainData2Unalign) / trainData1Unalign.shape[0]

    # plot the correlation coefficients
    if not cfg_dataset.equal_weights:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(corr_align)
        ax.plot(corr_unalign)
        ax.set_title("Correlation Coefficients of the Cross Covariance")
        ax.set_xlabel("Dimension of Eigenvalues")
        ax.set_ylabel("Correlation Coefficients")
        ax.legend(["Aligned", "Unaligned"])
        ax.set_ylim(0, 1)
        fig.savefig(cfg_dataset.paths.plots_path + "cca_corr.png")

    # plot ROC
    threshold_list = [i for i in np.linspace(-0.15, 0.65, 40).reshape(-1)]
    threshold_list += [-1, 1]
    threshold_list.sort()
    ROC_points_list = ROC_points(sim_align, sim_unalign, threshold_list)
    return ROC_points_list


def CLIP_like_data_align(cfg: DictConfig, shuffle_level: str = "dataset") -> list[tuple[float, float]]:
    """Align the audio and text data and calculate the similarity score using CLIP like models.

    Args:
        cfg: configuration file
        shuffle_level: shuffle level. It can be "dataset", "class", or "object".

    Returns:
        ROC points
    """
    # set random seed
    np.random.seed(cfg.seed)
    cfg_dataset, Data1, Data2 = load_CLIP_like_data(cfg)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Data1.shape[0])
    _, valData1 = train_test_split(Data1, trainIdx, valIdx)
    _, valData2 = train_test_split(Data2, trainIdx, valIdx)

    # copy data
    valData1Align = valData1.copy()
    valData2Align = valData2.copy()
    valData2Unalign = valData2.copy()

    valData2Unalign = shuffle_by_level(cfg_dataset, cfg.dataset, shuffle_level, valData2Unalign, trainIdx, valIdx)

    sim_align = cosine_sim(valData1Align, valData2Align)
    sim_unalign = cosine_sim(valData1Align, valData2Unalign)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(
        data_list=[sim_align, sim_unalign],
        legend=["Aligned", "Random shuffle"],
        title_str="Similarity Score Distribution",
        xlabel="Similarity Score",
        ylabel="Frequency",
        ax=ax,
    )
    save_fig(fig, cfg_dataset.paths.plots_path + f"cos_similarity_{shuffle_level}_CLIP_r{cfg.train_test_ratio}.png")

    # plot ROC
    threshold_list = [i for i in np.linspace(-1, 1, 40).reshape(-1)]
    threshold_list += [-1, 1]
    threshold_list.sort()
    ROC_points_list = ROC_points(sim_align, sim_unalign, threshold_list)
    return ROC_points_list


def shuffle_by_level(
    cfg_dataset: DictConfig,
    dataset: str,
    shuffle_level: str,
    valData2Unalign: np.ndarray,
    trainIdx: list[int],
    valIdx: list[int],
):
    """Shuffle the data by dataset, class, or object level.

    Args:
        cfg_dataset: configuration file
        dataset: dataset name
        shuffle_level: shuffle level. It can be "dataset", "class", or "object".
        valData2Unalign: unaligned data
        trainIdx: training indices
        valIdx: validation indices
    Returns:
        shuffled Data2
    """
    assert shuffle_level in ["dataset", "class", "object"], f"shuffle_level {shuffle_level} not supported."
    # sop shuffle by class or object
    if dataset == "sop":
        _, _, classes, obj_ids = load_SOP(cfg_dataset)
        if shuffle_level == "class":
            _, valClasses = train_test_split(classes, trainIdx, valIdx)
            # filter and shuffle data by classes or object ids
            val_class_dict_filter = filter_str_label(valClasses)
            valData2Unalign = shuffle_data_by_indices(valData2Unalign, val_class_dict_filter)

        elif shuffle_level == "object":
            _, valObjIds = train_test_split(obj_ids, trainIdx, valIdx)
            # filter and shuffle data by classes or object ids
            val_obj_dict_filter = filter_str_label(valObjIds)
            valData2Unalign = shuffle_data_by_indices(valData2Unalign, val_obj_dict_filter)

    if dataset != "sop" or shuffle_level == "dataset":
        np.random.shuffle(valData2Unalign)

    return valData2Unalign
