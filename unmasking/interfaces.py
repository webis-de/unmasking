from classifier import FeatureSet

from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import numpy

from abc import ABC, abstractmethod


class UnmaskingStrategy(ABC):
    """
    Base class for unmasking strategies.
    """
    
    def run(self, m: int, n: int, fs: FeatureSet, relative: bool = True):
        """
        Run ``m`` rounds of unmasking on given parameterized feature set.
        
        :param m: number of unmasking rounds
        :param n: number of features to use
        :param fs: parameterized feature set
        :param relative: whether to use relative (normalized) of absolute feature weights
        """
        train_matrix = []
        class_vec    = []
        if relative:
            it = fs.get_features_relative(n)
        else:
            it = fs.get_features_absolute(n)
        for row in it:
            l = len(row)
            train_matrix.append(row[0:l // 2])
            train_matrix.append(row[l // 2:l])
            # cls either "text 0" or "text 1" of a pair
            class_vec.append(0)
            class_vec.append(1)
        
        train_matrix = numpy.array(train_matrix)
        class_vec    = numpy.array(class_vec)
        clf = LinearSVC()
        for i in range(0, m):
            scores = cross_val_score(clf, train_matrix, class_vec, cv=10)
            print(scores.mean())
        
    @abstractmethod
    def transform(self, clf: LinearSVC) -> numpy.ndarray:
        """
        Transform the input vector according to the chosen unmasking strategy.
        
        :param clf: classifier used to discriminate the features
        :return: output feature vector
        """
        pass
