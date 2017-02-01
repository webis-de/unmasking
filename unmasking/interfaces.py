from classifier import FeatureSet

from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import numpy

from abc import ABC, abstractmethod


class UnmaskingStrategy(ABC):
    """
    Base class for unmasking strategies.
    """

    # noinspection PyPep8Naming
    def run(self, m: int, n: int, fs: FeatureSet, relative: bool = True, folds: int = 10):
        """
        Run ``m`` rounds of unmasking on given parameterized feature set.
        
        :param m: number of unmasking rounds
        :param n: number of features to use
        :param fs: parameterized feature set
        :param relative: whether to use relative (normalized) of absolute feature weights
        :param folds: number of cross-validation folds
        """
        X = []
        y = []
        if relative:
            it = fs.get_features_relative(n)
        else:
            it = fs.get_features_absolute(n)
        for row in it:
            l = len(row)
            X.append(row[0:l // 2])
            X.append(row[l // 2:l])
            
            # cls either "text 0" or "text 1" of a pair
            y.append(0)
            y.append(1)
        
        X = numpy.array(X)
        y = numpy.array(y)
        clf = LinearSVC()
        print(fs.cls)
        for i in range(0, m):
            clf.fit(X, y)
            scores = cross_val_score(clf, X, y, cv=folds)
            print(scores.mean())
            X = self.transform(X, clf)
    
    @abstractmethod
    def transform(self, data: numpy.ndarray, clf: LinearSVC) -> numpy.ndarray:
        """
        Transform the input tensor according to the chosen unmasking strategy.
        
        :param data: input rank-2 feature tensor of form [n_samples, n_features]
        :param clf: trained classifier used to discriminate the features
        :return: output feature tensor (may have contain different number of features,
                 but the number of samples must be the same)
        """
        pass
