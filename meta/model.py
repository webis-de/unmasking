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

from meta.interfaces import MetaClassificationModel

from concurrent.futures import ThreadPoolExecutor
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from typing import Any, Iterable

import asyncio
import numpy as np


# noinspection PyPep8Naming
class LinearMetaClassificationModel(MetaClassificationModel):
    """
    Meta classifier using a linear SVM model.
    """

    def __init__(self):
        super().__init__()
        self._threshold = 0.5

    # noinspection PyUnusedLocal
    def _get_estimator(self, index: int) -> BaseEstimator:
        return LinearSVC()

    async def fit(self, X: Iterable[Iterable[float]], y: Iterable[Any]):
        self.reset()

        # decision quality estimator
        self._clf.append(self._get_estimator(0))

        # sample classifier
        self._clf.append(self._get_estimator(1))

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            await asyncio.get_event_loop().run_in_executor(executor, self._clf[0].fit, X, y)

            # eliminate samples below threshold
            dist = self._clf[0].decision_function(X)
            X = np.fromiter((x for i, x in enumerate(X) if abs(dist[i]) >= self._threshold), dtype=[("", float), ("", float)])
            y = np.fromiter((y for i, y in enumerate(y) if abs(dist[i]) >= self._threshold), dtype=float)

            await asyncio.get_event_loop().run_in_executor(executor, self._clf[1].fit, X, y)
        finally:
            executor.shutdown()

    async def predict(self, X: Iterable[Iterable[float]]) -> Iterable[Any]:
        """
        Predict classes for samples in X
        If decision probability of a prediction is below the threshold, the array entry will be nan.
        """
        decision_func = self._clf[0].decision_function(X)
        pred = self._clf[1].predict(X)
        for i in range(len(decision_func)):
            if decision_func[i] < self._threshold:
                pred[i] = None
        return pred

    def decision_function(self, X: Iterable[Iterable[float]]) -> Iterable[float]:
        return self._clf[1].decision_function(X)

    @property
    def threshold(self) -> float:
        """Decision threshold below which to discard samples."""
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float):
        """Set decision threshold"""
        self._threshold = threshold
