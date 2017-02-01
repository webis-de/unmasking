from unmasking.interfaces import UnmaskingStrategy

import numpy
from sklearn.svm import LinearSVC


class FeatureRemoval(UnmaskingStrategy):
    """
    Classic feature removal as suggested by Koppel et al.
    """

    def __init__(self, num_eliminate: int):
        """
        :param num_eliminate: number of features to eliminate per iteration.
        """
        self._num_eliminate = num_eliminate

    def transform(self, data: numpy.ndarray, clf: LinearSVC) -> numpy.ndarray:
        if isinstance(clf.coef_, list):
            coef = numpy.absolute(clf.coef_[0])
        else:
            coef = numpy.absolute(clf.coef_)
        
        for i in range(0, self._num_eliminate):
            if data.shape[1] == 0:
                return data
            
            index = numpy.argmax(coef)
            coef  = numpy.delete(coef, index)
            data  = numpy.delete(data, index, 1)
        
        return data
