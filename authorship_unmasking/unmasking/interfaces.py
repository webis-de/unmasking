# Copyright (C) 2017-2019 Janek Bevendorff, Webis Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from authorship_unmasking.features.interfaces import FeatureSet
from authorship_unmasking.event.dispatch import EventBroadcaster, MultiProcessEventContext
from authorship_unmasking.event.events import UnmaskingTrainCurveEvent
from authorship_unmasking.job.interfaces import Strategy

from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
import numpy

from abc import ABCMeta, abstractmethod
import sys
from typing import List
import warnings


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
        self._use_mean_coefs = True

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
    def use_mean_coefs(self) -> bool:
        """Whether to use mean feature coefficients for vector transformation."""
        return self._use_mean_coefs

    @use_mean_coefs.setter
    def use_mean_coefs(self, use_mean_coefs: bool):
        """Set whether to use mean coefficients"""
        self._use_mean_coefs = use_mean_coefs

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
        if not buffer:
            print('WARNING: Curve buffering is turned off.', file=sys.stderr),
            print('         Set "buffer_curves" to true in your job config for better performance.\n', file=sys.stderr)
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
            if MultiProcessEventContext().terminate_event.is_set():
                return

            try:
                cv = cross_validate(clf, X, y, cv=self._folds, return_estimator=True, return_train_score=False)
                score = max(0.0, (cv['test_score'].mean() - .5) * 2)
                cv_models = cv["estimator"]

                if self._monotonize:
                    values.append(score)
                else:
                    values.append(score)
                    event.values = values

                if len(cv_models[0].coef_.shape) > 1:
                    coef = numpy.array([c.coef_[0] for c in cv_models])
                else:
                    coef = numpy.array([c.coef_ for c in cv_models])

                if self._use_mean_coefs:
                    coef = numpy.mean(coef, axis=0)

                if not self._monotonize and not self._buffer_curves:
                    await EventBroadcaster().publish("onUnmaskingRoundFinished", event, self.__class__)
                    event = UnmaskingTrainCurveEvent.new_event(event)

                if i < self._iterations - 1:
                    X = await self.transform(X, coef)
                    if X.size == 0:
                        # Nothing to do anymore
                        break
            except ValueError:
                continue

        if self._monotonize:
            event.values = self._do_monotonize(values)
        if self._monotonize or self._buffer_curves:
            await EventBroadcaster().publish("onUnmaskingRoundFinished", event, self.__class__)
        event = UnmaskingTrainCurveEvent.new_event(event)
        await EventBroadcaster().publish("onUnmaskingFinished", event, self.__class__)

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
    async def transform(self, data: numpy.ndarray, coefs: numpy.ndarray) -> numpy.ndarray:
        """
        Transform the input tensor according to the chosen unmasking strategy.
        
        :param data: input feature matrix of shape (m_samples, n_features)
        :param coefs: trained feature coefficients for each CV fold (of shape (k_folds, n_features)
                      or (1, n_features) if use_mean_coefs is True)
        :return: output feature tensor (may have contain different number of features,
                 but the number of samples must be the same)
        """
        pass
