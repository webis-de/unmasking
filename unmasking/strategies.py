from unmasking.interfaces import UnmaskingStrategy


from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC


class FeatureRemoval(UnmaskingStrategy):
    """
    Classic feature removal as suggested by Koppel et al.
    """

    def __init__(self, num_features: int):
        """
        :param num_features: number of features to remove.
        """
        self._num_features = num_features

    def transform(self, vectorizer: DictVectorizer, clf: LinearSVC) -> DictVectorizer:
        pass
