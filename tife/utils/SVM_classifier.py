import numpy as np
from sklearn import svm


class SVM_classifier:
    def __init__(self, 
                 X_train: np.ndarray, 
                 y_train: np.ndarray,) -> None:
        '''A class of SVM classifier
        Input:
            X_train: training data (n_samples, n_features)
            y_train: training labels
            X_test: testing data
            y_test: testing labels
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.classifier = svm.SVC()
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_train)
        self.train_accuracy = self.get_accuracy(y_pred=self.y_pred, y_test=self.y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        '''Predict labels
        Input:
            X_test: testing data
        Returns:
            predicted labels
        '''
        return self.classifier.predict(X_test)

    def get_accuracy(self, y_pred: np.ndarray, y_test: np.ndarray) -> float:
        '''Get accuracy
        Input:
            y_pred: predicted labels
            y_test: true labels

        Returns:
            accuracy
        '''
        return np.mean(y_pred == y_test)