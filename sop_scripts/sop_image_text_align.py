import pickle

import matplotlib.pyplot as plt
import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig
from swarm_visualizer.histogram import (
    plot_several_pdf,
)
from swarm_visualizer.utility.general_utils import save_fig

from mmda.utils.data_utils import (
    get_train_test_split_index,
    origin_centered,
    train_test_split,
)
from mmda.utils.hydra_utils import hydra_main
from mmda.utils.sim_utils import ROC_points, cosine_sim, weighted_corr_sim


@hydra_main(version_base=None, config_path="../config", config_name="sop")
def main(cfg: DictConfig):  # noqa: D103
    SOP_align(cfg)
    SOP_CLIP_align(cfg)
    return


def SOP_align(cfg: DictConfig):  # noqa: D103
    # set random seed
    np.random.seed(cfg.seed)

    # load image embeddings and text embeddings
    with open(cfg.paths.save_path + f"data/SOP_img_emb_{cfg.img_encoder}.pkl", "rb") as f:
        Img = pickle.load(f)
    with open(cfg.paths.save_path + f"data/SOP_text_emb_{cfg.text_encoder}.pkl", "rb") as f:
        Txt = pickle.load(f)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Img.shape[0])
    trainImg, valImg = train_test_split(Img, trainIdx, valIdx)
    trainTxt, valTxt = train_test_split(Txt, trainIdx, valIdx)

    ### aligned case: not shuffle the data
    trainImgAlign, valImgAlign = trainImg.copy(), valImg.copy()
    trainTxtAlign, valTxtAlign = trainTxt.copy(), valTxt.copy()
    # zero mean data
    trainImgAlign, trainImg_mean = origin_centered(trainImgAlign)
    trainTxtAlign, trainTxt_mean = origin_centered(trainTxtAlign)
    valImgAlign = valImgAlign - trainImg_mean  # zero mean
    valTxtAlign = valTxtAlign - trainTxt_mean  # zero mean
    # make sure the data is zero mean
    assert np.allclose(
        trainImgAlign.mean(axis=0), 0, atol=1e-4
    ), f"trainImgAlign not zero mean: {trainImgAlign.mean(axis=0)}"
    assert np.allclose(
        trainTxtAlign.mean(axis=0), 0, atol=1e-4
    ), f"trainTxtAlign not zero mean: {trainTxtAlign.mean(axis=0)}"

    # CCA dimensionality reduction
    img_text_CCA = CCA(latent_dimensions=cfg.CCA_dim)
    trainImgAlign, trainTxtAlign = img_text_CCA.fit_transform((trainImgAlign, trainTxtAlign))
    if cfg.equal_weights:
        corr_align = np.ones((trainTxtAlign.shape[1],))  # dim,
    else:
        corr_align = np.diag(trainImgAlign.T @ trainTxtAlign) / trainImgAlign.shape[0]  # dim,

    # calculate the similarity score
    valImgAlign, valTxtAlign = img_text_CCA.transform((valImgAlign, valTxtAlign))
    sim_align = weighted_corr_sim(valImgAlign, valTxtAlign, corr_align, dim=cfg.sim_dim)

    ### unaligned case: shuffle the data
    # shuffle only the text data
    trainImgUnalign, valImgUnalign = trainImg.copy(), valImg.copy()
    trainTxtUnalign, valTxtUnalign = trainTxt.copy(), valTxt.copy()
    np.random.shuffle(trainTxtUnalign)
    np.random.shuffle(valTxtUnalign)
    # zero mean data
    trainImgUnalign, trainImg_mean_ = origin_centered(trainImgUnalign)
    trainTxtUnalign, trainTxt_mean_ = origin_centered(trainTxtUnalign)
    valImgUnalign = valImgUnalign - trainImg_mean_  # zero mean
    valTxtUnalign = valTxtUnalign - trainTxt_mean_  # zero mean
    # make sure the data is zero mean
    assert np.allclose(
        trainImgUnalign.mean(axis=0), 0, atol=1e-4
    ), f"trainImgUnalign not zero mean: {trainImgUnalign.mean(axis=0)}"
    assert np.allclose(
        trainTxtUnalign.mean(axis=0), 0, atol=1e-4
    ), f"trainTxtUnalign not zero mean: {trainTxtUnalign.mean(axis=0)}"

    valImgAlign, valTxtAlign = img_text_CCA.transform((valImgUnalign, valTxtUnalign))
    sim_unalign = weighted_corr_sim(valImgAlign, valTxtAlign, corr_align, dim=cfg.sim_dim)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(
        data_vector_list=[sim_align, sim_unalign],
        legend=["Aligned", "Unaligned"],
        title_str="Similarity Score Distribution",
        xlabel="Similarity Score",
        ylabel="Frequency",
        ax=ax,
    )
    if cfg.equal_weights:
        save_fig(
            fig,
            cfg.paths.plots_path + f"similarity_score_dataset_r{cfg.train_test_ratio}_dim{cfg.sim_dim}_noweight.png",
        )
    else:
        save_fig(fig, cfg.paths.plots_path + f"similarity_score_dataset_r{cfg.train_test_ratio}_dim{cfg.sim_dim}.png")

    # CCA dimensionality reduction
    img_text_CCA_unalign = CCA(latent_dimensions=cfg.CCA_dim)
    trainImgUnalign, trainTxtUnalign = img_text_CCA_unalign.fit_transform((trainImgUnalign, trainTxtUnalign))
    corr_unalign = np.diag(trainImgUnalign.T @ trainTxtUnalign) / trainImgUnalign.shape[0]

    # plot the correlation coefficients
    if not cfg.equal_weights:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(corr_align)
        ax.plot(corr_unalign)
        ax.set_title("Correlation Coefficients of the Cross Covariance")
        ax.set_xlabel("Dimension of Eigenvalues")
        ax.set_ylabel("Correlation Coefficients")
        ax.legend(["Aligned", "Unaligned"])
        ax.set_ylim(0, 1)
        fig.savefig(cfg.paths.plots_path + "cca_corr.png")

    # plot ROC
    threshold_list = [i for i in np.linspace(-0.15, 0.65, 25).reshape(-1)]
    threshold_list += [-1, 1]
    threshold_list.sort()
    ROC_points_list = ROC_points(sim_align, sim_unalign, threshold_list)
    # auc = cal_AUC(ROC_points_list)
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.plot([x[0] for x in ROC_points_list], [x[1] for x in ROC_points_list], 'o-')
    # ax.legend([f'AUC: {auc:.3f}'])
    # ax.set_title('ROC Curve')
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # fig.savefig(cfg.paths.plots_path + f'ROC_dataset_dim{cfg.sim_dim}_{cfg.train_test_ratio}.png')

    return ROC_points_list


def SOP_CLIP_align(cfg):  # noqa: D103
    # set random seed
    np.random.seed(cfg.seed)

    # load image embeddings and text embeddings
    with open(cfg.paths.save_path + "data/SOP_img_emb_clip.pkl", "rb") as f:
        Img = pickle.load(f)
    with open(cfg.paths.save_path + "data/SOP_text_emb_clip.pkl", "rb") as f:
        Txt = pickle.load(f)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Img.shape[0])
    _, valImg = train_test_split(Img, trainIdx, valIdx)
    _, valTxt = train_test_split(Txt, trainIdx, valIdx)

    # copy data
    valImgAlign = valImg.copy()
    valTxtAlign = valTxt.copy()
    valTxtUnalign = valTxt.copy()
    np.random.shuffle(valTxtUnalign)

    sim_align = cosine_sim(valImgAlign, valTxtAlign)
    sim_unalign = cosine_sim(valImgAlign, valTxtUnalign)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(
        data_vector_list=[sim_align, sim_unalign],
        legend=["Aligned", "Class level shuffle"],
        title_str="Similarity Score Distribution",
        xlabel="Similarity Score",
        ylabel="Frequency",
        ax=ax,
    )
    save_fig(fig, cfg.paths.plots_path + f"cos_similarity_dataset_r{cfg.train_test_ratio}_CLIP.png")

    # plot ROC
    threshold_list = [i for i in np.linspace(-0.8, 0.8, 30).reshape(-1)]
    threshold_list += [-1, 1]
    threshold_list.sort()
    ROC_points_list = ROC_points(sim_align, sim_unalign, threshold_list)
    return ROC_points_list


if __name__ == "__main__":
    main()
