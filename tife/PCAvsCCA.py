import pickle

import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from utils.linear_algebra_utils import origin_centered
from utils.SVM_classifier import SVM_classifier

import hydra


@hydra.main(version_base=None, config_path='config', config_name='main_config')
def main(cfg: DictConfig):
    print("Img encoder:", cfg.img_encoder, "Text encoder:", cfg.text_encoder)
    # load waterbirds embeddings
    with open(cfg.save_dir + f'data/waterbird_img_emb_train_test_{cfg.img_encoder}.pkl', 'rb') as f:
        dbFeatImg = pickle.load(f)
    with open(cfg.save_dir + f'data/waterbird_img_emb_val_{cfg.img_encoder}.pkl', 'rb') as f:
        qFeatImg = pickle.load(f)
    with open(cfg.save_dir + f'data/waterbird_text_emb_train_test_{cfg.text_encoder}.pkl', 'rb') as f:
        dbFeatText = pickle.load(f)
    with open(cfg.save_dir + f'data/waterbird_text_emb_val_{cfg.text_encoder}.pkl', 'rb') as f:
        qFeatText = pickle.load(f)

    # zero-mean data
    dbFeatImg, dbFeatImg_mean = origin_centered(dbFeatImg)
    dbFeatText, dbFeatText_mean = origin_centered(dbFeatText)
    qFeatImg = qFeatImg - dbFeatImg_mean
    qFeatText = qFeatText - dbFeatText_mean
    # qFeatImg, _ = origin_centered(qFeatImg)
    # qFeatText, _ = origin_centered(qFeatText)
    # print(dbFeatImg.shape, dbFeatText.shape, qFeatImg.shape, qFeatText.shape)

    # make sure the data is zero mean
    assert np.allclose(dbFeatImg.mean(axis=0), 0, atol=1e-4), f"dbFeatImg not zero mean: {dbFeatImg.mean(axis=0)}"
    assert np.allclose(dbFeatText.mean(axis=0), 0, atol=1e-4), f"dbFeatText not zero mean: {dbFeatText.mean(axis=0)}"

    # load ground truth
    with open(cfg.save_dir + 'data/waterbird_gt_train_test.pkl', 'rb') as f:
        gt_train_test = pickle.load(f)
    gt_train_test = np.array([x[1] for x in gt_train_test])
    with open(cfg.save_dir + 'data/waterbird_gt_val.pkl', 'rb') as f:
        gt_val = pickle.load(f)
    gt_val = np.array([_[1] for _ in gt_val])
    with open(cfg.save_dir + 'data/waterbird_bg_gt_train_test.pkl', 'rb') as f:
        bg_gt_train_test = pickle.load(f)
    bg_gt_train_test = np.array([_[1] for _ in bg_gt_train_test])
    with open(cfg.save_dir + 'data/waterbird_bg_gt_val.pkl', 'rb') as f:
        bg_gt_val = pickle.load(f)
    bg_gt_val = np.array([_[1] for _ in bg_gt_val])

    if cfg.gt_category == 'bg':
        gt_train_test = bg_gt_train_test
        gt_val = bg_gt_val
    
    assert gt_train_test.shape[0] == dbFeatImg.shape[0], f"gt_train_test shape {gt_train_test.shape} != dbFeatImg shape {dbFeatImg.shape}"
    assert gt_val.shape[0] == qFeatImg.shape[0], f"gt_val shape {gt_val.shape} != qFeatImg shape {qFeatImg.shape}"

    # PCA and CCA
    for dim in range(50, 701, 50):
        print("Embedding Dimension:", dim)

        # fit CCA and PCA
        img_text_CCA = CCA(latent_dimensions=dim)
        img_text_CCA.fit((dbFeatImg, dbFeatText))
        imgPca = PCA(n_components=dim)
        imgPca.fit(dbFeatImg)

        # calculate PCA
        pca_dbFeatImg = imgPca.transform(dbFeatImg)[:,:dim]
        pca_qFeatImg = imgPca.transform(qFeatImg)[:,:dim]
        
        # fit PCA SVM and calculate accuracy
        svm = SVM_classifier(pca_dbFeatImg, gt_train_test)
        qPred = svm.predict(pca_qFeatImg)
        dbAcc = svm.get_accuracy(svm.y_pred, gt_train_test)
        qAcc = svm.get_accuracy(qPred, gt_val)
        print("PCA train img SVM Accuracy {:.4f} | PCA val SVM Accuracy {:.4f}".format(dbAcc, qAcc))

        # calculate CCA
        cca_dbFeatImg, cca_dbFeatText = img_text_CCA.transform((dbFeatImg, dbFeatText))
        cca_qFeatImg, cca_qFeatText = img_text_CCA.transform((qFeatImg, qFeatText))

        # fit SVM
        svm = SVM_classifier(cca_dbFeatImg, gt_train_test)
        qPred = svm.predict(cca_qFeatImg)
        dbAcc = svm.get_accuracy(svm.y_pred, gt_train_test)
        qAcc = svm.get_accuracy(qPred, gt_val)
        print("CCA train img SVM Accuracy {:.4f} | CCA val SVM Accuracy {:.4f}".format(dbAcc, qAcc))

        # fit SVM
        svm = SVM_classifier(cca_dbFeatText, gt_train_test)
        qPred = svm.predict(cca_qFeatText)
        dbAcc = svm.get_accuracy(svm.y_pred, gt_train_test)
        qAcc = svm.get_accuracy(qPred, gt_val)
        print("CCA train txt SVM Accuracy {:.4f} | CCA val txt SVM Accuracy {:.4f}".format(dbAcc, qAcc))


if __name__ == '__main__':
    main()
