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

from conf.interfaces import Configurable
from input.interfaces import SamplePair

from abc import ABCMeta, abstractmethod
import numpy
from typing import Iterable, Tuple


class ChunkSampler(Configurable, metaclass=ABCMeta):
    """
    Base class for chunk samplers used for generating pairs of chunks from :class:`SamplePair`s.

    Chunk sampler properties with setters defined via @property.setter
    can be set at runtime via job configuration.
    """
    
    @abstractmethod
    def generate_chunk_pairs(self, pair: SamplePair) -> Iterable[Tuple[str, str]]:
        """
        Generate pairs of chunks from the given :class:`SamplePair.

        :param pair: text pair to create chunk pairs from
        :return: generator or other iterable producing the chunk pairs
        """
        pass


class FeatureSet(Configurable, metaclass=ABCMeta):
    """
    Base class for text discrimination feature sets.

    Feature properties with setters defined via @property.setter
    can be set at runtime via job configuration.
    """
    
    def __init__(self, pair: SamplePair = None, sampler: ChunkSampler = None):
        """
        :param pair: pair of chunked texts
        :param sampler: :class:`ChunkSampler` for sampling chunks from ``pair``
        """
        self._pair = pair
        self._sampler = sampler
    
    @property
    def pair(self) -> SamplePair:
        """Pair from which this feature set has been generated."""
        return self._pair

    @pair.setter
    def pair(self, pair):
        self._pair = pair

    @property
    def chunk_sampler(self) -> ChunkSampler:
        """Chunk sampler"""
        return self._sampler

    @chunk_sampler.setter
    def chunk_sampler(self, sampler):
        self._sampler = sampler

    @abstractmethod
    def get_features_absolute(self, n: int) -> Iterable[numpy.ndarray]:
        """
        Create feature vectors from the chunked text pair.
        Each feature vector will have length ``n`` per chunk, resulting in an overall size
        of 2``n'` for each chunk pair.

        :param n: dimension of the feature vector to create for each chunk
        :return: generator or iterable of 2n-dimensional feature vectors
        """
        pass
    
    @abstractmethod
    def get_features_relative(self, n: int) -> Iterable[numpy.ndarray]:
        """
        Create feature vectors from the chunked text pair with relative (normalized) feature weights.
        Each feature vector will have length ``n`` per chunk, resulting in an overall size
        of 2``n'` for each chunk pair.

        :param n: dimension of the feature vector to create for each chunk
        :return: generator or iterable of 2n-dimensional feature vectors
        """
        pass
