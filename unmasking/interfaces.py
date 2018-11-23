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

from features.interfaces import FeatureSet
from event.dispatch import EventBroadcaster, MultiProcessEventContext
from event.events import UnmaskingTrainCurveEvent
from job.interfaces import Strategy

from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import numpy

from abc import ABCMeta, abstractmethod
from typing import List


class UnmaskingStrategy(Strategy, metaclass=ABCMeta):
    """
    Base class for unmasking execution strategies.
    
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
        self._buffer_curves = True
        self._values = []
        self._iterations = 10
        self._vector_size = 250
        self._relative = False
        self._folds = 10
        self._monotonize = False

    @property
    def iterations(self) -> int:
        """Number of unmasking iterations."""
        return self._iterations

    @iterations.setter
    def iterations(self, iterations: int):
        """Set number of unmasking iterations."""
        self._iterations = iterations

    @property
    def vector_size(self) -> int:
        """Feature vector size."""
        return self._vector_size

    @vector_size.setter
    def vector_size(self, vector_size: int):
        """Set feature vector size."""
        self._vector_size = vector_size

    @property
    def relative(self) -> bool:
        """Whether to use relative or absolute feature weights."""
        return self._relative

    @relative.setter
    def relative(self, relative: bool):
        """Set whether to use relative or absolute feature weights."""
        self._relative = relative

    @property
    def folds(self) -> int:
        """Number of cross-validation folds to use for discriminating feature vectors."""
        return self._folds

    @folds.setter
    def folds(self, folds: int):
        """Set number of cross-validation folds to use for discriminating feature vectors."""
        self._folds = folds

    @property
    def monotonize(self) -> bool:
        """Whether to monotonize curves."""
        return self._monotonize

    @monotonize.setter
    def monotonize(self, monotonize: bool):
        """Set whether to monotonize curves."""
        self._monotonize = monotonize

    @property
    def buffer_curves(self) -> bool:
        """Whether to buffer curves."""
        return self._buffer_curves

    @buffer_curves.setter
    def buffer_curves(self, buffer: bool):
        """Set whether to buffer curves. Set to False to send update events after each round."""
        self._buffer_curves = buffer

    # noinspection PyPep8Naming
    async def run(self, fs: FeatureSet):
        """
        Run ``m`` rounds of unmasking on given parametrized feature set.

        :param fs: parametrized feature set to run unmasking on
        """
        clf = LinearSVC()
        X = []
        y = []

        if self._relative:
            it = fs.get_features_relative(self._vector_size)
        else:
            it = fs.get_features_absolute(self._vector_size)
        for row in it:
            l = len(row)
            X.append(row[0:l // 2])
            X.append(row[l // 2:l])

            # cls either "text 0" or "text 1" of a pair
            y.append(0)
            y.append(1)

        X = numpy.array(X)
        y = numpy.array(y)
        group_id = UnmaskingTrainCurveEvent.generate_group_id([self.__class__.__name__ + ":" + fs.pair.pair_id])
        event = UnmaskingTrainCurveEvent(group_id, 0, self._iterations, fs.pair, fs.__class__)
        values = []
        for i in range(self._iterations):
            if MultiProcessEventContext.terminate_event.is_set():
                return

            try:
                clf.fit(X, y)
                scores = cross_val_score(clf, X, y, cv=self._folds)
                score = max(0.0, (scores.mean() - .5) * 2)

                if self._monotonize:
                    values.append(score)
                else:
                    values.append(score)
                    event.values = values

                if isinstance(clf.coef_, list):
                    coef = numpy.array(clf.coef_[0])
                else:
                    coef = numpy.array(clf.coef_)

                if not self._monotonize and not self._buffer_curves:
                    await EventBroadcaster.publish("onUnmaskingRoundFinished", event, self.__class__)
                    event = UnmaskingTrainCurveEvent.new_event(event)

                if i < self._iterations - 1:
                    X = await self.transform(X, coef)
                    if not X:
                        # Nothing to do anymore
                        break
            except ValueError:
                continue

        if self._monotonize:
            event.values = self._do_monotonize(values)
        if self._monotonize or self._buffer_curves:
            await EventBroadcaster.publish("onUnmaskingRoundFinished", event, self.__class__)
        event = UnmaskingTrainCurveEvent.new_event(event)
        await EventBroadcaster.publish("onUnmaskingFinished", event, self.__class__)

    def _do_monotonize(self, values: List[float]):
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
