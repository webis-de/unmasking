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

from features.feature_sets import FeatureSet
from unmasking.interfaces import UnmaskingStrategy

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
