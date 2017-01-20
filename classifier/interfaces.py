from classifier.chunking import SamplePair, ChunkSampler

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Iterable


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
    def get_features_absolute(self, n: int) -> Iterable[Tuple[Dict[str, float], SamplePair.Class]]:
        """
        Create feature vectors from the chunked text pair.
        Each feature vector will have length ``n`` per chunk, resulting in an overall size
        of 2``n'` for each chunk pair.
        
        The feature vectors are represented as Python dicts containing {key => value pairs} of
        {feature_name => feature_point}.

        :param n: dimension of the feature vector to create for each chunk
        :return: generator or iterable of tuples of 2n-dimensional feature vectors (dicts) and their classes
        """
        pass
    
    @abstractmethod
    def get_features_relative(self, n: int) -> Iterable[Tuple[Dict[str, float], SamplePair.Class]]:
        """
        Create feature vectors from the chunked text pair with relative (normalized) feature weights.
        Each feature vector will have length ``n`` per chunk, resulting in an overall size
        of 2``n'` for each chunk pair.
        
        The feature vectors are represented as Python dicts containing {key => value pairs} of
        {feature_name => feature_point}.

        :param n: dimension of the feature vector to create for each chunk
        :return: generator or iterable of tuples of 2n-dimensional feature vectors (dicts) and their classes
        """
        pass
