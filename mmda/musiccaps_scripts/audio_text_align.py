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
from mmda.utils.sim_utils import (
    ROC_points,
    cal_AUC,
    cosine_sim,
    weighted_corr_sim,
)


@hydra_main(version_base=None, config_path='../config', config_name='musiccaps')
def main(cfg: DictConfig):
    MusicCaps_align(cfg)
    # MusicCaps_CLAP_align(cfg)
    return

def MusicCaps_align(cfg: DictConfig):
    # set random seed
    np.random.seed(cfg.seed)

    # load image embeddings and text embeddings
    with open(cfg.paths.save_path + f'MusicCaps_audio_emb_{cfg.audio_encoder}.pkl', 'rb') as f:
        Audio = pickle.load(f)
    with open(cfg.paths.save_path + f'MusicCaps_text_emb_{cfg.text_encoder}.pkl', 'rb') as f:
        Txt = pickle.load(f)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Audio.shape[0], cfg.seed)
    trainAudio, valAudio = train_test_split(Audio, trainIdx, valIdx)
    trainTxt, valTxt = train_test_split(Txt, trainIdx, valIdx)

    ### aligned case: not shuffle the data
    trainAudioAlign, valAudioAlign = trainAudio.copy(), valAudio.copy()
    trainTxtAlign, valTxtAlign = trainTxt.copy(), valTxt.copy()
    # zero mean data
    trainAudioAlign, trainAudio_mean = origin_centered(trainAudioAlign)
    trainTxtAlign, trainTxt_mean = origin_centered(trainTxtAlign)
    valAudioAlign = valAudioAlign - trainAudio_mean # zero mean
    valTxtAlign = valTxtAlign - trainTxt_mean # zero mean
    # make sure the data is zero mean
    assert np.allclose(trainAudioAlign.mean(axis=0), 0, atol=1e-4), f"trainAudioAlign not zero mean: {trainAudioAlign.mean(axis=0)}"
    assert np.allclose(trainTxtAlign.mean(axis=0), 0, atol=1e-4), f"trainTxtAlign not zero mean: {trainTxtAlign.mean(axis=0)}"

    # CCA dimensionality reduction
    img_text_CCA = CCA(latent_dimensions=cfg.CCA_dim)
    trainAudioAlign, trainTxtAlign = img_text_CCA.fit_transform((trainAudioAlign, trainTxtAlign))
    if cfg.equal_weights:
        corr_align = np.ones((trainTxtAlign.shape[1],)) # dim,
    else:
        corr_align = np.diag(trainAudioAlign.T @ trainTxtAlign) / trainAudioAlign.shape[0] # dim,

    # calculate the similarity score
    valAudioAlign, valTxtAlign = img_text_CCA.transform((valAudioAlign, valTxtAlign))
    sim_align = weighted_corr_sim(valAudioAlign, valTxtAlign, corr_align, dim=cfg.sim_dim)

    ### unaligned case: shuffle the data
    # shuffle only the text data
    trainAudioUnalign, valAudioUnalign = trainAudio.copy(), valAudio.copy()
    trainTxtUnalign, valTxtUnalign = trainTxt.copy(), valTxt.copy()
    np.random.shuffle(trainTxtUnalign)
    np.random.shuffle(valTxtUnalign)
    # zero mean data
    trainAudioUnalign, trainAudio_mean_ = origin_centered(trainAudioUnalign)
    trainTxtUnalign, trainTxt_mean_ = origin_centered(trainTxtUnalign)
    valAudioUnalign = valAudioUnalign - trainAudio_mean_ # zero mean
    valTxtUnalign = valTxtUnalign - trainTxt_mean_ # zero mean
    # make sure the data is zero mean
    assert np.allclose(trainAudioUnalign.mean(axis=0), 0, atol=1e-4), f"trainAudioUnalign not zero mean: {trainAudioUnalign.mean(axis=0)}"
    assert np.allclose(trainTxtUnalign.mean(axis=0), 0, atol=1e-4), f"trainTxtUnalign not zero mean: {trainTxtUnalign.mean(axis=0)}"

    valAudioAlign, valTxtAlign = img_text_CCA.transform((valAudioUnalign, valTxtUnalign))
    sim_unalign = weighted_corr_sim(valAudioAlign, valTxtAlign, corr_align, dim=cfg.sim_dim)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(data_vector_list=[sim_align, sim_unalign], 
                     legend=['Aligned', 'Unaligned'], 
                     title_str='Similarity Score Distribution', 
                     xlabel='Similarity Score', 
                     ylabel='Frequency', 
                     ax=ax)
    if cfg.equal_weights:
        save_fig(fig, cfg.paths.plots_path + f'similarity_score_dataset_dim{cfg.sim_dim}_{cfg.train_test_ratio}_noweight.png')
    else:
        save_fig(fig, cfg.paths.plots_path + f'similarity_score_dataset_dim{cfg.sim_dim}_{cfg.train_test_ratio}.png')

    # CCA dimensionality reduction
    img_text_CCA_unalign = CCA(latent_dimensions=cfg.CCA_dim)
    trainAudioUnalign, trainTxtUnalign = img_text_CCA_unalign.fit_transform((trainAudioUnalign, trainTxtUnalign))
    corr_unalign = np.diag(trainAudioUnalign.T @ trainTxtUnalign) / trainAudioUnalign.shape[0]
    
    # plot the correlation coefficients
    if not cfg.equal_weights:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(corr_align)
        ax.plot(corr_unalign)
        ax.set_title('Correlation Coefficients of the Cross Covariance')
        ax.set_xlabel('Dimension of Eigenvalues')
        ax.set_ylabel('Correlation Coefficients')
        ax.legend(['Aligned', 'Unaligned'])
        ax.set_ylim(0, 1)
        fig.savefig(cfg.paths.plots_path + 'cca_corr.png')

    # plot ROC
    threshold_list = [i for i in np.linspace(-0.15, 0.65, 40).reshape(-1)]
    threshold_list += [-1, 1]
    threshold_list.sort()
    ROC_points_list = ROC_points(sim_align, sim_unalign, threshold_list)
    auc = cal_AUC(ROC_points_list)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([x[0] for x in ROC_points_list], [x[1] for x in ROC_points_list], 'o-')
    ax.legend([f'AUC: {auc:.3f}'])
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(cfg.paths.plots_path + f'ROC_class_dim{cfg.sim_dim}_{cfg.train_test_ratio}.png')
    
    return ROC_points_list

def MusicCaps_CLAP_align(cfg):
    # set random seed
    np.random.seed(cfg.seed)

    # load image embeddings and text embeddings
    with open(cfg.paths.save_path + 'MusicCaps_audio_emb_clap.pkl', 'rb') as f:
        Audio = pickle.load(f)
    with open(cfg.paths.save_path + 'MusicCaps_text_emb_clap.pkl', 'rb') as f:
        Txt = pickle.load(f)

    trainIdx, valIdx = get_train_test_split_index(cfg.train_test_ratio, Audio.shape[0], cfg.seed)
    _, valAudio = train_test_split(Audio, trainIdx, valIdx)
    _, valTxt = train_test_split(Txt, trainIdx, valIdx)

    # copy data
    valAudioAlign =  valAudio.copy()
    valTxtAlign = valTxt.copy()
    valTxtUnalign = valTxt.copy()
    np.random.shuffle(valTxtUnalign)

    sim_align = cosine_sim(valAudioAlign, valTxtAlign)
    sim_unalign = cosine_sim(valAudioAlign, valTxtUnalign)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_several_pdf(data_vector_list=[sim_align, sim_unalign], 
                     legend=['Aligned', 'Random shuffle'], 
                     title_str='Similarity Score Distribution', 
                     xlabel='Similarity Score', 
                     ylabel='Frequency', 
                     ax=ax)
    save_fig(fig, cfg.paths.plots_path + f'cos_similarity_dataset_CLAP_{cfg.train_test_ratio}.png')

    # plot ROC
    threshold_list = [i for i in np.linspace(-1, 1, 40).reshape(-1)]
    threshold_list += [-1, 1]
    threshold_list.sort()
    ROC_points_list = ROC_points(sim_align, sim_unalign, threshold_list)
    return ROC_points_list


if __name__ == '__main__':
    main()
