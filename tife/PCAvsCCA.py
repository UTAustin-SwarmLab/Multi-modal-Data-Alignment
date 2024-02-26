import numpy as np
import pickle
from tqdm import tqdm
from cca_zoo.linear import CCA
from sklearn.decomposition import PCA
from utils.SVM_classifier import SVM_classifier
from utils.linear_algebra_utils import origin_centered

img_encoder = "clip"
text_encoder = "clip"

root_dir =  "/home/pl22767/Data/waterbird_landfraction50_forest2water2/"
save_dir = "/nas/tirr/waterbird/"

# load waterbirds embeddings
with open(save_dir + f'data/waterbird_img_emb_train_test_{img_encoder}.pkl', 'rb') as f:
    dbFeatImg = pickle.load(f)
with open(save_dir + f'data/waterbird_img_emb_val_{img_encoder}.pkl', 'rb') as f:
    qFeatImg = pickle.load(f)
with open(save_dir + f'data/waterbird_text_emb_train_test_{text_encoder}.pkl', 'rb') as f:
    dbFeatText = pickle.load(f)
with open(save_dir + f'data/waterbird_text_emb_val_{text_encoder}.pkl', 'rb') as f:
    qFeatText = pickle.load(f)

# zero-mean data
dbFeatImg, dbFeatImg_mean = origin_centered(dbFeatImg)
dbFeatText, dbFeatText_mean = origin_centered(dbFeatText)
qFeatImg = qFeatImg - dbFeatImg_mean
qFeatText = qFeatText - dbFeatText_mean
# qFeatImg, _ = origin_centered(qFeatImg)
# qFeatText, _ = origin_centered(qFeatText)
print(dbFeatImg.shape, dbFeatText.shape, qFeatImg.shape, qFeatText.shape)
print(dbFeatImg_mean.shape, dbFeatText_mean.shape)

# load ground truth
with open(save_dir + 'data/waterbird_gt_train_test.pkl', 'rb') as f:
    gt_train_test = pickle.load(f)
gt_train_test = np.array([x[1] for x in gt_train_test])
with open(save_dir + 'data/waterbird_gt_val.pkl', 'rb') as f:
    gt_val = pickle.load(f)
gt_val = np.array([_[1] for _ in gt_val])


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
    print("PCA train SVM Accuracy", dbAcc, "PCA val SVM Accuracy", qAcc)

    # calculate CCA
    cca_dbFeatImg, cca_dbFeatText = img_text_CCA.transform((dbFeatImg, dbFeatText))
    cca_qFeatImg, cca_qFeatText = img_text_CCA.transform((qFeatImg, qFeatText))

    # fit SVM
    svm = SVM_classifier(cca_dbFeatImg, gt_train_test)
    qPred = svm.predict(cca_qFeatImg)
    dbAcc = svm.get_accuracy(svm.y_pred, gt_train_test)
    qAcc = svm.get_accuracy(qPred, gt_val)
    print("CCA train SVM Accuracy", dbAcc, "CCA val SVM Accuracy", qAcc)