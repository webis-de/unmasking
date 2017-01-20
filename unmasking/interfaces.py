from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer

from abc import ABC, abstractmethod


class UnmaskingStrategy(ABC):
    """
    Base class for unmasking strategies.
    """
    
    @abstractmethod
    def transform(self, vectorizer: DictVectorizer, clf: LinearSVC) -> DictVectorizer:
        """
        Transform the input vector according to the chosen unmasking strategy.
        
        :param vectorizer: feature vectorizer
        :param clf: classifier used to discriminate the features
        :return: output feature vector
        """
        pass
