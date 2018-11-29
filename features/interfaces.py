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
