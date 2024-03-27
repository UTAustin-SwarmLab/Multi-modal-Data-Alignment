import pickle

import numpy as np
from cca_zoo.linear import CCA
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from utils.SVM_classifier import SVM_classifier

import hydra
from tife.utils.data_utils import origin_centered


@hydra.main(version_base=None, config_path='config', config_name='main_config')
def main(cfg: DictConfig):
    print("Img encoder:", cfg.img_encoder, "| Text encoder:", cfg.text_encoder)
    print("Ground truth category:", cfg.gt_category)
    # set random seed
    np.random.seed(cfg.seed)

    # load waterbirds text embeddings + image embeddings 
    if cfg.imbal == "imbal95":
        gt_cat = "_cat" # or ""
        with open(cfg.save_dir + f'data/waterbird_imbal95_img_emb_train_{cfg.img_encoder}.pkl', 'rb') as f:
            trainFeatImg = pickle.load(f)
        with open(cfg.save_dir + f'data/waterbird_imbal95_img_emb_val_{cfg.img_encoder}.pkl', 'rb') as f:
            valFeatImg = pickle.load(f)
        with open(cfg.save_dir + f'data/waterbird_imbal95_text_emb_train_{cfg.text_encoder}.pkl', 'rb') as f:
            trainFeatText = pickle.load(f)
        with open(cfg.save_dir + f'data/waterbird_imbal95_text_emb_val_{cfg.text_encoder}.pkl', 'rb') as f:
            valFeatText = pickle.load(f)
        # load ground truth
        with open(cfg.save_dir + f'data/waterbird_imbal95{gt_cat}_gt_train.pkl', 'rb') as f:
            gt_train = pickle.load(f)
        gt_train = np.array([x[1] for x in gt_train])
        with open(cfg.save_dir + f'data/waterbird_imbal95{gt_cat}_gt_val.pkl', 'rb') as f:
            gt_val = pickle.load(f)
        gt_val = np.array([_[1] for _ in gt_val])
        print("gt_train:", gt_train[-5:], "gt_val:", gt_val[-5:])
    else: 
        with open(cfg.save_dir + f'data/waterbird_img_emb_train_test_{cfg.img_encoder}.pkl', 'rb') as f:
            trainFeatImg = pickle.load(f)
        with open(cfg.save_dir + f'data/waterbird_img_emb_val_{cfg.img_encoder}.pkl', 'rb') as f:
            valFeatImg = pickle.load(f)
        with open(cfg.save_dir + f'data/waterbird_text_emb_train_test{cfg.gt_category}_{cfg.text_encoder}.pkl', 'rb') as f:
            trainFeatText = pickle.load(f)
        with open(cfg.save_dir + f'data/waterbird_text_emb_val{cfg.gt_category}_{cfg.text_encoder}.pkl', 'rb') as f:
            valFeatText = pickle.load(f)
        # load ground truth
        with open(cfg.save_dir + f'data/waterbird{cfg.gt_category}_gt_train.pkl', 'rb') as f:
            gt_train = pickle.load(f)
        gt_train = np.array([x[1] for x in gt_train])
        with open(cfg.save_dir + f'data/waterbird{cfg.gt_category}_gt_val.pkl', 'rb') as f:
            gt_val = pickle.load(f)
        gt_val = np.array([_[1] for _ in gt_val])

    # zero-mean data
    trainFeatImg, trainFeatImg_mean = origin_centered(trainFeatImg)
    trainFeatText, trainFeatText_mean = origin_centered(trainFeatText)
    valFeatImg = valFeatImg - trainFeatImg_mean
    valFeatText = valFeatText - trainFeatText_mean
    print("trainFeatImg shape:", trainFeatImg.shape, "trainFeatText shape:", trainFeatText.shape)
    print("valFeatImg shape:", valFeatImg.shape, "valFeatText shape:", valFeatText.shape)

    # make sure the data is zero mean
    assert np.allclose(trainFeatImg.mean(axis=0), 0, atol=1e-4), f"trainFeatImg not zero mean: {trainFeatImg.mean(axis=0)}"
    assert np.allclose(trainFeatText.mean(axis=0), 0, atol=1e-4), f"trainFeatText not zero mean: {trainFeatText.mean(axis=0)}"

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
    svm = SVM_classifier(trainFeatText, gt_train)
    valPred = svm.predict(valFeatText)
    trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
    valAcc = svm.get_accuracy(valPred, gt_val)
    print("text SVM Accuracy {:.4f} | val Accuracy {:.4f}".format(trainAcc, valAcc))

    # img + text
    svm = SVM_classifier(np.concatenate((trainFeatImg, trainFeatText), axis=1), gt_train)
    valPred = svm.predict(np.concatenate((valFeatImg, valFeatText), axis=1))
    trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
    valAcc = svm.get_accuracy(valPred, gt_val)
    print("img+txt SVM Accuracy {:.4f} | val img+txt Accuracy {:.4f}".format(trainAcc, valAcc))

    # PCA and CCA
    for dim in range(100, 701, 100):
        print("Embedding Dimension:", dim)

        # fit CCA and PCA
        img_text_CCA = CCA(latent_dimensions=dim)
        img_text_CCA.fit((trainFeatImg, trainFeatText))
        imgPca = PCA(n_components=dim)
        imgPca.fit(trainFeatImg)
        textPca = PCA(n_components=dim)
        textPca.fit(trainFeatText)
        img_text_PCA = PCA(n_components=dim)
        img_text_PCA.fit(np.concatenate((trainFeatImg, trainFeatText), axis=1))

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
        pca_trainFeatText = textPca.transform(trainFeatText)
        pca_valFeatText = textPca.transform(valFeatText)

        # fit PCA SVM and calculate accuracy for text
        svm = SVM_classifier(pca_trainFeatText, gt_train)
        valPred = svm.predict(pca_valFeatText)
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("PCA train txt SVM Accuracy {:.4f} | PCA val txt SVM Accuracy {:.4f}".format(trainAcc, valAcc))

        # fit PCA SVM and calculate accuracy for img + text
        svm = SVM_classifier(img_text_PCA.transform(np.concatenate((trainFeatImg, trainFeatText), axis=1)), gt_train)
        valPred = svm.predict(img_text_PCA.transform(np.concatenate((valFeatImg, valFeatText), axis=1)))
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("PCA train img+txt SVM Accuracy {:.4f} | PCA val img+txt SVM Accuracy {:.4f}".format(trainAcc, valAcc))

        # calculate CCA
        cca_trainFeatImg, cca_trainFeatText = img_text_CCA.transform((trainFeatImg, trainFeatText))
        cca_valFeatImg, cca_valFeatText = img_text_CCA.transform((valFeatImg, valFeatText))

        # fit SVM
        svm = SVM_classifier(cca_trainFeatImg, gt_train)
        valPred = svm.predict(cca_valFeatImg)
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("CCA train img SVM Accuracy {:.4f} | CCA val SVM Accuracy {:.4f}".format(trainAcc, valAcc))

        # fit SVM
        svm = SVM_classifier(cca_trainFeatText, gt_train)
        valPred = svm.predict(cca_valFeatText)
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("CCA train txt SVM Accuracy {:.4f} | CCA val txt SVM Accuracy {:.4f}".format(trainAcc, valAcc))

        # Img + text CCA 
        svm = SVM_classifier(np.concatenate((cca_trainFeatImg, cca_trainFeatText), axis=1), gt_train)
        valPred = svm.predict(np.concatenate((cca_valFeatImg, cca_valFeatText), axis=1))
        trainAcc = svm.get_accuracy(svm.y_pred, gt_train)
        valAcc = svm.get_accuracy(valPred, gt_val)
        print("CCA train img+txt SVM Accuracy {:.4f} | CCA val img+txt SVM Accuracy {:.4f}".format(trainAcc, valAcc))


if __name__ == '__main__':
    main()
