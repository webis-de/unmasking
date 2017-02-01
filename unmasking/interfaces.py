from classifier import FeatureSet
from event.events import UnmaskingTrainCurveEvent

from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import numpy

from abc import ABC, abstractmethod


class UnmaskingStrategy(ABC):
    """
    Base class for unmasking strategies.
    
    Events published by this class:
    
    * `onUnmaskingRoundFinished`: [type: UnmaskingTrainCurveEvent]
                                  fired whenever a single round of unmasking has finished
                                  to update accuracy curves
    """
    
    def __init__(self):
        """
        Initialize unmasking strategy. LinearSVC() is used as default estimator.
        """
        self._clf = LinearSVC()
    
    @property
    def clf(self):
        """Get estimator."""
        return self._clf
    
    @clf.setter
    def clf(self, clf):
        """
        Set estimator.
        Must implement ``fit()`` and provide an attribute ``coef_`` after fitting data, which
        contains the trained feature coefficients.
        """
        if not hasattr(clf, "fit"):
            raise ValueError("Estimator does not implement fit()")
        self._clf = clf
    
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
        event = UnmaskingTrainCurveEvent(m, fs.pair)
        for i in range(0, m):
            self._clf.fit(X, y)
            scores = cross_val_score(self._clf, X, y, cv=folds)
            event.values = scores.mean()
            if isinstance(self._clf.coef_, list):
                coef = numpy.array(self._clf.coef_[0])
            else:
                coef = numpy.array(self._clf.coef_)
            X = self.transform(X, coef)
    
    @abstractmethod
    def transform(self, data: numpy.ndarray, coef: numpy.ndarray) -> numpy.ndarray:
        """
        Transform the input tensor according to the chosen unmasking strategy.
        
        :param data: input rank-2 feature tensor of form [n_samples, n_features]
        :param coef: trained feature coefficients
        :return: output feature tensor (may have contain different number of features,
                 but the number of samples must be the same)
        """
        pass
