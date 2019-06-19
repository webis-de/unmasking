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

from authorship_unmasking.features.feature_sets import FeatureSet
from authorship_unmasking.unmasking.interfaces import UnmaskingStrategy

import numpy
from typing import Union


class FeatureRemoval(UnmaskingStrategy):
    """
    Classic feature removal as suggested by Koppel et al.
    """

    def __init__(self, num_eliminate: int = 10):
        """
        :param num_eliminate: number of features to eliminate per iteration.
        """
        super().__init__()
        self._num_eliminate = num_eliminate

    @property
    def eliminate(self) -> int:
        """Get number of eliminations per round"""
        return self._num_eliminate

    @eliminate.setter
    def eliminate(self, eliminate: Union[int, str]):
        """Set number of eliminations per round"""
        self._num_eliminate = eliminate

    async def run(self, fs: FeatureSet):
        if self._iterations == "auto":
            self._iterations = self._vector_size // self._num_eliminate
        await super().run(fs)

    async def transform(self, data: numpy.ndarray, coefs: numpy.ndarray) -> numpy.ndarray:
        for i in range(self._num_eliminate):
            if i < self._num_eliminate / 2:
                indices = numpy.argmax(coefs) if self.use_mean_coefs else numpy.argmax(numpy.max(coefs, axis=0))
            else:
                indices = numpy.argmin(coefs) if self.use_mean_coefs else numpy.argmin(numpy.min(coefs, axis=0))
            coefs = numpy.delete(coefs, indices) if self.use_mean_coefs else numpy.delete(coefs, indices, axis=1)
            data = numpy.delete(data, indices, axis=1)

        return data
