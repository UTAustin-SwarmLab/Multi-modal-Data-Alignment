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

import hydra
from tife.utils.data_utils import (
    get_train_test_split_index,
    origin_centered,
    train_test_split,
)
from tife.utils.sim_utils import weighted_corr_sim


@hydra.main(version_base=None, config_path='config', config_name='sop')
def main(cfg: DictConfig):
    # set random seed
    np.random.seed(cfg.seed)
    plots_folder_path = os.path.join(os.path.dirname(__file__), "./plots/")

    # load image embeddings and text embeddings
    with open(cfg.save_dir + f'data/SOP_img_emb_{cfg.img_encoder}.pkl', 'rb') as f:
        Img = pickle.load(f)
    with open(cfg.save_dir + f'data/SOP_text_emb_{cfg.text_encoder}.pkl', 'rb') as f:
        Txt = pickle.load(f)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Img.shape[0], cfg.seed)
    trainImg, valImg = train_test_split(Img, trainIdx, valIdx)
    trainTxt, valTxt = train_test_split(Txt, trainIdx, valIdx)

    # ### aligned case: not shuffle the data
    trainImgAlign, valImgAlign = trainImg.copy(), valImg.copy()
    trainTxtAlign, valTxtAlign = trainTxt.copy(), valTxt.copy()
    # zero mean data
    trainImgAlign, trainImg_mean = origin_centered(trainImgAlign)
    trainTxtAlign, trainTxt_mean = origin_centered(trainTxtAlign)
    valImgAlign = valImgAlign - trainImg_mean # zero mean
    valTxtAlign = valTxtAlign - trainTxt_mean # zero mean
    # make sure the data is zero mean
    assert np.allclose(trainImgAlign.mean(axis=0), 0, atol=1e-4), f"trainImgAlign not zero mean: {trainImgAlign.mean(axis=0)}"
    assert np.allclose(trainTxtAlign.mean(axis=0), 0, atol=1e-4), f"trainTxtAlign not zero mean: {trainTxtAlign.mean(axis=0)}"

    # CCA dimensionality reduction
    img_text_CCA = CCA(latent_dimensions=700)
    trainImgAlign, trainTxtAlign = img_text_CCA.fit_transform((trainImgAlign, trainTxtAlign))
    corr_align = np.diag(trainImgAlign.T @ trainTxtAlign) / trainImgAlign.shape[0] # dim, 1
    print(corr_align[:10])

    # calculate the similarity score
    valImgAlign, valTxtAlign = img_text_CCA.transform((valImgAlign, valTxtAlign))
    sim_align = weighted_corr_sim(valImgAlign, valTxtAlign, corr_align, dim=cfg.sim_dim)
    print(f"Aligned case: similarity score: {sim_align.mean()}")


    ### unaligned case: shuffle the data
    # shuffle only the text data
    trainImgUnalign, valImgUnalign = trainImg.copy(), valImg.copy()
    trainTxtUnalign, valTxtUnalign = trainTxt.copy(), valTxt.copy()
    np.random.shuffle(trainTxtUnalign)
    np.random.shuffle(valTxtUnalign)
    # zero mean data
    trainImgUnalign, trainImg_mean_ = origin_centered(trainImgUnalign)
    trainTxtUnalign, trainTxt_mean_ = origin_centered(trainTxtUnalign)
    valImgUnalign = valImgUnalign - trainImg_mean_ # zero mean
    valTxtUnalign = valTxtUnalign - trainTxt_mean_ # zero mean
    # make sure the data is zero mean
    assert np.allclose(trainImgUnalign.mean(axis=0), 0, atol=1e-4), f"trainImgUnalign not zero mean: {trainImgUnalign.mean(axis=0)}"
    assert np.allclose(trainTxtUnalign.mean(axis=0), 0, atol=1e-4), f"trainTxtUnalign not zero mean: {trainTxtUnalign.mean(axis=0)}"

    valImgAlign, valTxtAlign = img_text_CCA.transform((valImgUnalign, valTxtUnalign))
    sim_unalign = weighted_corr_sim(valImgAlign, valTxtAlign, corr_align, dim=cfg.sim_dim)
    print(f"Unaligned case: similarity score: {sim_unalign.mean()}")
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(data_vector_list=[sim_align, sim_unalign], 
                     legend=['Aligned', 'Unaligned'], 
                     title_str='Similarity Score Distribution', 
                     xlabel='Similarity Score', 
                     ylabel='Frequency', 
                     ax=ax)
    save_fig(fig, plots_folder_path + 'similarity_score_aligned_vs_unaligned.png')

    # CCA dimensionality reduction
    img_text_CCA_unalign = CCA(latent_dimensions=700)
    trainImgUnalign, trainTxtUnalign = img_text_CCA_unalign.fit_transform((trainImgUnalign, trainTxtUnalign))
    corr_unalign = np.diag(trainImgUnalign.T @ trainTxtUnalign) / trainImgUnalign.shape[0]
    
    # plot the covariance matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(corr_align)
    ax.plot(corr_unalign)
    ax.set_title('Correlation Coefficients of the Cross Covariance')
    ax.set_xlabel('Dimension of Eigenvalues')
    ax.set_ylabel('Correlation Coefficients')
    ax.legend(['Aligned', 'Unaligned'])
    ax.set_ylim(0, 1)
    fig.savefig(plots_folder_path + 'cca_corr_coeff_unaligned.png')

    
if __name__ == '__main__':
    main()
