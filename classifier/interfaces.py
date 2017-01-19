from classifier.chunking import SamplePair, ChunkSampler

from abc import ABC, abstractmethod
from typing import List, Tuple, Iterable


class FeatureSet(ABC):
    """
    Base class for text discrimination feature sets.
    """
    
    def __init__(self, pair: SamplePair, sampler: ChunkSampler):
        """
        :param pair: pair of chunked texts
        :param sampler: :class:`ChunkSampler` for sampling chunks from ``pair``
        """
        self._pair = pair
        self._sampler = sampler
    
    @abstractmethod
    def get_features_absolute(self, n: int) -> Iterable[Tuple[List[float], SamplePair.Class]]:
        """
        Create feature vectors from the chunked text pair.

        :param n: dimension of the feature vector to create
        :return: generator or iterable of n-dimensional feature vectors and their classes
        """
        pass
    
    @abstractmethod
    def get_features_relative(self, n: int) -> Iterable[Tuple[List[float], SamplePair.Class]]:
        """
        Create feature vectors from the chunked text pair with relative (normalized) feature weights.

        :param n: dimension of the feature vector to create
        :return: generator or iterable of n-dimensional feature vectors and their classes
        """
        pass
