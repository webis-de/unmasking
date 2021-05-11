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

from authorship_unmasking.meta.interfaces import MetaClassificationModel

from concurrent.futures import ThreadPoolExecutor
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
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

    def _get_estimator(self) -> BaseEstimator:
        return LinearSVC()

    async def fit(self, X: Iterable[Iterable[float]], y: Iterable[Any]):
        self.reset()
        self._clf = self.get_configured_estimator()

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            await asyncio.get_event_loop().run_in_executor(executor, self._clf.fit, X, y)
        finally:
            executor.shutdown()

    async def optimize(self, X: Iterable[Iterable[float]], y: Iterable[int]):
        """
        Optimize model parameters using cross-validated grid search
        """
        estimator = self._get_estimator()
        parameters = {
            "C": [0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10],
            "loss": ["hinge", "squared_hinge"],
            "class_weight": [None, "balanced", {0: 1, 1: 2}, {0: 2, 1: 1}]
        }
        grid = GridSearchCV(estimator, parameters, cv=min(5, *np.bincount(np.array(y, int))), n_jobs=-1)
        grid.fit(X, y)
        self._clf_params = grid.best_estimator_.get_params()

    async def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """
        Predict classes for samples in X.
        If decision probability of a prediction is below the threshold, the array entry will be -1.
        """
        decision_func = self._clf.decision_function(X)
        pred = self._clf.predict(X).astype(int)
        for i in range(len(decision_func)):
            if abs(decision_func[i]) < self._threshold:
                pred[i] = -1

        return pred

    async def decision_function(self, X: Iterable[Iterable[float]]) -> Iterable[float]:
        return self._clf.decision_function(X)

    @property
    def threshold(self) -> float:
        """Decision threshold below which to discard samples."""
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float):
        """Set decision threshold"""
        self._threshold = threshold
