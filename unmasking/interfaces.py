# General-purpose unmasking framework
# Copyright (C) 2017 Janek Bevendorff, Webis Group
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

from classifier.interfaces import FeatureSet
from event.dispatch import EventBroadcaster, MultiProcessEventContext
from event.events import UnmaskingTrainCurveEvent
from conf.interfaces import Configurable
from input.interfaces import SamplePair

from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import numpy

from abc import ABC, abstractmethod
from typing import List


class UnmaskingStrategy(ABC, Configurable):
    """
    Base class for unmasking strategies.
    
    Events published by this class:
    
    * `onUnmaskingRoundFinished`: [type: UnmaskingTrainCurveEvent]
                                  fired whenever a single round of unmasking has finished
                                  to update accuracy curves
    * `onUnmaskingFinished`:      [type: UnmaskingTrainCurveEvent]
                                  fired when unmasking curve generation for a text has finished
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
    async def run(self, pair: SamplePair, m: int, n: int, fs: FeatureSet, relative: bool = False,
                  folds: int = 10, monotonize: bool = False):
        """
        Run ``m`` rounds of unmasking on given parametrized feature set.

        :param pair: input pair from which to generate this curve
        :param m: number of unmasking rounds
        :param n: number of features to use
        :param fs: parametrized feature set
        :param relative: whether to use relative (normalized) of absolute feature weights
        :param folds: number of cross-validation folds
        :param monotonize: whether to monotonize curves (i.e., no point will be larger than the previous point)
        """
        clf = LinearSVC()
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
        group_id = UnmaskingTrainCurveEvent.generate_group_id([self.__class__.__name__ + ":" + pair.pair_id])
        event = UnmaskingTrainCurveEvent(group_id, 0, m, fs.pair, fs.__class__)
        values = []
        for i in range(m):
            if MultiProcessEventContext.terminate_event.is_set():
                return

            try:
                clf.fit(X, y)
                scores = cross_val_score(clf, X, y, cv=folds)
                score = max(0.0, (scores.mean() - .5) * 2)

                if monotonize:
                    values.append(score)
                else:
                    values.append(score)
                    event.values = values

                if isinstance(clf.coef_, list):
                    coef = numpy.array(clf.coef_[0])
                else:
                    coef = numpy.array(clf.coef_)

                if not monotonize:
                    await EventBroadcaster.publish("onUnmaskingRoundFinished", event, self.__class__)
                    event = UnmaskingTrainCurveEvent.new_event(event)

                if i < m - 1:
                    X = await self.transform(X, coef)
            except ValueError:
                continue

        if monotonize:
            event.values = self._monotonize(values)
            await EventBroadcaster.publish("onUnmaskingRoundFinished", event, self.__class__)
        event = UnmaskingTrainCurveEvent.new_event(event)
        await EventBroadcaster.publish("onUnmaskingFinished", event, self.__class__)

    def _monotonize(self, values: List[float]):
        # monotonize from the left
        values_l = numpy.zeros(len(values))
        prev_value = 1.0
        for i, v in enumerate(values):
            values_l[i] = min(prev_value, v)
            prev_value = values_l[i]

        # monotonize from the right
        values_r = numpy.zeros(len(values))
        prev_value = 0.0
        for i in range(len(values) - 1, -1, -1):
            values_r[i] = max(prev_value, values[i])
            prev_value = values_r[i]

        # calculate squared differences to find the better of both approximations
        values_arr = numpy.array(values)
        delta_l = numpy.sum(numpy.square(values_arr - values_l))
        delta_r = numpy.sum(numpy.square(values_arr - values_r))

        if delta_l <= delta_r:
            return list(values_l)
        return list(values_r)

    @abstractmethod
    async def transform(self, data: numpy.ndarray, coef: numpy.ndarray) -> numpy.ndarray:
        """
        Transform the input tensor according to the chosen unmasking strategy.
        
        :param data: input rank-2 feature tensor of form [n_samples, n_features]
        :param coef: trained feature coefficients
        :return: output feature tensor (may have contain different number of features,
                 but the number of samples must be the same)
        """
        pass
