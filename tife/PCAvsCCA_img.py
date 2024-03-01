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
    print("Ground truth category:", cfg.gt_category)
    # load waterbirds image embeddings 
    if cfg.imbal == "imbal95":
        gt_cat = "_cat" # or ""
        with open(cfg.save_dir + 'data/waterbird_imbal95_img_emb_train_clip.pkl', 'rb') as f:
            trainFeatImg = pickle.load(f)
        with open(cfg.save_dir + 'data/waterbird_imbal95_img_emb_val_clip.pkl', 'rb') as f:
            valFeatImg = pickle.load(f)
        with open(cfg.save_dir + 'data/waterbird_imbal95_img_emb_train_dino.pkl', 'rb') as f:
            trainFeatImg2 = pickle.load(f)
        with open(cfg.save_dir + 'data/waterbird_imbal95_img_emb_val_dino.pkl', 'rb') as f:
            valFeatImg2 = pickle.load(f)
        # load ground truth
        with open(cfg.save_dir + f'data/waterbird_imbal95{gt_cat}_gt_train.pkl', 'rb') as f:
            gt_train = pickle.load(f)
        gt_train = np.array([x[1] for x in gt_train])
        with open(cfg.save_dir + f'data/waterbird_imbal95{gt_cat}_gt_val.pkl', 'rb') as f:
            gt_val = pickle.load(f)
        gt_val = np.array([_[1] for _ in gt_val])
        print("gt_train:", gt_train[-5:], "gt_val:", gt_val[-5:])
    else: 
        with open(cfg.save_dir + 'data/waterbird_img_emb_train_test_clip.pkl', 'rb') as f:
            trainFeatImg = pickle.load(f)
        with open(cfg.save_dir + 'data/waterbird_img_emb_val_clip.pkl', 'rb') as f:
            valFeatImg = pickle.load(f)
        with open(cfg.save_dir + 'data/waterbird_img_emb_train_test_dino.pkl', 'rb') as f:
            trainFeatImg2 = pickle.load(f)
        with open(cfg.save_dir + 'data/waterbird_img_emb_val_dino.pkl', 'rb') as f:
            valFeatImg2 = pickle.load(f)

    # zero-mean data
    trainFeatImg, trainFeatImg_mean = origin_centered(trainFeatImg)
    trainFeatImg2, trainFeatImg2_mean = origin_centered(trainFeatImg2)
    valFeatImg = valFeatImg - trainFeatImg_mean
    valFeatImg2 = valFeatImg2 - trainFeatImg2_mean
    print("trainFeatImg shape:", trainFeatImg.shape, "trainFeatImg2 shape:", trainFeatImg2.shape)
    print("valFeatImg shape:", valFeatImg.shape, "valFeatImg2 shape:", valFeatImg2.shape)

    # make sure the data is zero mean
    assert np.allclose(trainFeatImg.mean(axis=0), 0, atol=1e-4), f"trainFeatImg not zero mean: {trainFeatImg.mean(axis=0)}"
    assert np.allclose(trainFeatImg2.mean(axis=0), 0, atol=1e-4), f"trainFeatImg2 not zero mean: {trainFeatImg2.mean(axis=0)}"

    assert gt_train.shape[0] == trainFeatImg.shape[0], f"gt_train shape {gt_train.shape} != trainFeatImg shape {trainFeatImg.shape}"
    assert gt_val.shape[0] == valFeatImg.shape[0], f"gt_val shape {gt_val.shape} != valFeatImg shape {valFeatImg.shape}"

    # maximum performance of SVM
    # image only
    svm = SVM_classifier(trainFeatImg, gt_train)
    valPred = svm.predict(valFeatImg)
    trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
    valAcc = svm.get_accuracy(valPred, gt_val)
    print("img SVM Accuracy {:.4f} | val Accuracy {:.4f}".format(trainAcc, valAcc))

    # text only
    svm = SVM_classifier(trainFeatImg2, gt_train)
    valPred = svm.predict(valFeatImg2)
    trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
    valAcc = svm.get_accuracy(valPred, gt_val)
    print("img2 SVM Accuracy {:.4f} | val Accuracy {:.4f}".format(trainAcc, valAcc))

    # img + text
    svm = SVM_classifier(np.concatenate((trainFeatImg, trainFeatImg2), axis=1), gt_train)
    valPred = svm.predict(np.concatenate((valFeatImg, valFeatImg2), axis=1))
    trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
    valAcc = svm.get_accuracy(valPred, gt_val)
    print("img 1+2 SVM Accuracy {:.4f} | val img+txt Accuracy {:.4f}".format(trainAcc, valAcc))

    # PCA and CCA
    for dim in range(100, 701, 100):
        print("Embedding Dimension:", dim)

        # fit CCA and PCA
        img_text_CCA = CCA(latent_dimensions=dim)
        img_text_CCA.fit((trainFeatImg, trainFeatImg2))
        imgPca = PCA(n_components=dim)
        imgPca.fit(trainFeatImg)
        textPca = PCA(n_components=dim)
        textPca.fit(trainFeatImg2)
        img_text_PCA = PCA(n_components=dim)
        img_text_PCA.fit(np.concatenate((trainFeatImg, trainFeatImg2), axis=1))

        # calculate img PCA
        pca_trainFeatImg = imgPca.transform(trainFeatImg)
        pca_valFeatImg = imgPca.transform(valFeatImg)
        
        # fit PCA SVM and calculate accuracy for img
        svm = SVM_classifier(pca_trainFeatImg, gt_train)
        valPred = svm.predict(pca_valFeatImg)
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("PCA train img SVM Accuracy {:.4f} | PCA val SVM Accuracy {:.4f}".format(trainAcc, valAcc))

        # calculate text PCA
        pca_trainFeatImg2 = textPca.transform(trainFeatImg2)
        pca_valFeatImg2 = textPca.transform(valFeatImg2)

        # fit PCA SVM and calculate accuracy for text
        svm = SVM_classifier(pca_trainFeatImg2, gt_train)
        valPred = svm.predict(pca_valFeatImg2)
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("PCA train img 2 SVM Accuracy {:.4f} | PCA val txt SVM Accuracy {:.4f}".format(trainAcc, valAcc))

        # fit PCA SVM and calculate accuracy for img + text
        svm = SVM_classifier(img_text_PCA.transform(np.concatenate((trainFeatImg, trainFeatImg2), axis=1)), gt_train)
        valPred = svm.predict(img_text_PCA.transform(np.concatenate((valFeatImg, valFeatImg2), axis=1)))
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("PCA train img 1+2 SVM Accuracy {:.4f} | PCA val img+txt SVM Accuracy {:.4f}".format(trainAcc, valAcc))

        # calculate CCA
        cca_trainFeatImg, cca_trainFeatImg2 = img_text_CCA.transform((trainFeatImg, trainFeatImg2))
        cca_valFeatImg, cca_valFeatImg2 = img_text_CCA.transform((valFeatImg, valFeatImg2))

        # fit SVM
        svm = SVM_classifier(cca_trainFeatImg, gt_train)
        valPred = svm.predict(cca_valFeatImg)
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("CCA train img SVM Accuracy {:.4f} | CCA val SVM Accuracy {:.4f}".format(trainAcc, valAcc))

        # fit SVM
        svm = SVM_classifier(cca_trainFeatImg2, gt_train)
        valPred = svm.predict(cca_valFeatImg2)
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("CCA train img 2 SVM Accuracy {:.4f} | CCA val txt SVM Accuracy {:.4f}".format(trainAcc, valAcc))

        # Img + text CCA 
        svm = SVM_classifier(np.concatenate((cca_trainFeatImg, cca_trainFeatImg2), axis=1), gt_train)
        valPred = svm.predict(np.concatenate((cca_valFeatImg, cca_valFeatImg2), axis=1))
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("CCA train img 1+2 SVM Accuracy {:.4f} | CCA val img+txt SVM Accuracy {:.4f}".format(trainAcc, valAcc))


if __name__ == '__main__':
    main()
