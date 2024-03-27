import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig
from swarm_visualizer.histogram import (
    plot_several_pdf,
)
from swarm_visualizer.utility.general_utils import save_fig

from tife.utils.data_utils import (
    filter_str_label,
    get_train_test_split_index,
    load_SOP,
    origin_centered,
    shuffle_data_by_indices,
    train_test_split,
)
from tife.utils.hydra_utils import hydra_main
from tife.utils.sim_utils import ROC_points, weighted_corr_sim


@hydra_main(version_base=None, config_path='config', config_name='sop')
def SOP_obj_align(cfg: DictConfig):
    # set random seed
    np.random.seed(cfg.seed)
    plots_folder_path = os.path.join(os.path.dirname(__file__), "./plots/")

    # load raw data
    _, _, classes, obj_ids = load_SOP(cfg)

    # load image embeddings and text embeddings
    with open(cfg.save_dir + f'data/SOP_img_emb_{cfg.img_encoder}.pkl', 'rb') as f:
        Img = pickle.load(f)
    with open(cfg.save_dir + f'data/SOP_text_emb_{cfg.text_encoder}.pkl', 'rb') as f:
        Txt = pickle.load(f)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Img.shape[0], cfg.seed)
    trainImg, valImg = train_test_split(Img, trainIdx, valIdx)
    trainTxt, valTxt = train_test_split(Txt, trainIdx, valIdx)
    trainClasses, valClasses = train_test_split(classes, trainIdx, valIdx)
    trainObjIds, valObjIds = train_test_split(obj_ids, trainIdx, valIdx)

    # origin centered
    trainImg, trainImgMean = origin_centered(trainImg)
    trainTxt, trainTxtMean = origin_centered(trainTxt)
    valImg = valImg - trainImgMean
    valTxt = valTxt - trainTxtMean
    # make sure the data is zero mean
    assert np.allclose(trainImg.mean(axis=0), 0, atol=1e-4), f"trainImg not zero mean: {trainImg.mean(axis=0)}"
    assert np.allclose(trainTxt.mean(axis=0), 0, atol=1e-4), f"trainTxt not zero mean: {trainTxt.mean(axis=0)}"

    # copy data
    trainImgAlign, valImgAlign = trainImg.copy(), valImg.copy()
    trainTxtAlign, valTxtAlign = trainTxt.copy(), valTxt.copy()
    _, valImgUnalign = trainImg.copy(), valImg.copy()
    _, valTxtUnalign = trainTxt.copy(), valTxt.copy()

    # filter and shuffle data by classes or object ids
    val_obj_dict_filter = filter_str_label(valObjIds)
    valTxtUnalign = shuffle_data_by_indices(valTxtUnalign, val_obj_dict_filter, seed=cfg.seed)
    assert not np.allclose(valTxtUnalign, valTxt, atol=1e-4), "valTxtUnalign not shuffled correctly"
    assert np.allclose(valTxtUnalign.mean(axis=0), valTxt.mean(axis=0), atol=1e-4), "valTxtUnalign not zero mean"

    # CCA dimensionality reduction
    img_text_CCA = CCA(latent_dimensions=cfg.sim_dim)
    trainImgAlign, trainTxtAlign = img_text_CCA.fit_transform((trainImgAlign, trainTxtAlign))
    if cfg.equal_weights:
        corr = np.ones((trainTxtAlign.shape[1],)) # dim,
    else:
        corr = np.diag(trainImgAlign.T @ trainTxtAlign) / trainImgAlign.shape[0] # dim,

    # calculate the similarity score
    valImgAlign, valTxtAlign = img_text_CCA.transform((valImgAlign, valTxtAlign))
    sim_align = weighted_corr_sim(valImgAlign, valTxtAlign, corr, dim=cfg.sim_dim)

    # calculate the similarity score
    valImgUnalign, valTxtUnalign = img_text_CCA.transform((valImgUnalign, valTxtUnalign))
    sim_unalign = weighted_corr_sim(valImgUnalign, valTxtUnalign, corr, dim=cfg.sim_dim)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(data_vector_list=[sim_align, sim_unalign], 
                     legend=['Aligned', 'Object level shuffle'], 
                     title_str='Similarity Score Distribution', 
                     xlabel='Similarity Score', 
                     ylabel='Frequency', 
                     ax=ax)
    if cfg.equal_weights:
        save_fig(fig, plots_folder_path + f'similarity_score_obj_dim{cfg.sim_dim}_{cfg.train_test_ratio}_noweight.png')
    else:
        save_fig(fig, plots_folder_path + f'similarity_score_obj_dim{cfg.sim_dim}_{cfg.train_test_ratio}.png')

    # plot ROC
    threshold_list = [i for i in np.linspace(-0.15, 0.65, 20).reshape(-1)]
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
    # fig.savefig(plots_folder_path + f'ROC_obj_dim{cfg.sim_dim}_{cfg.train_test_ratio}.png')

    return ROC_points_list

if __name__ == '__main__':
    SOP_obj_align()
