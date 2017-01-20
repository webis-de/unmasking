from abc import ABC, abstractmethod
from typing import List

from classifier.chunking import SamplePair


class CorpusParser(ABC):
    """
    Base class for corpus parsers.
    """
    
    class CorpusParserIterator(ABC):
        """
        Iterator class for :class:`CorpusParser`.
        """
        
        def __init__(self, parser):
            """
            :type parser: CorpusParser.CorpusParserIterator
            """
            self.parser = parser
        
        @abstractmethod
        def __next__(self) -> SamplePair:
            pass
    
    def __init__(self, corpus_path: str, chunk_size: int, language: str = "english", cache_size: int = 400):
        """
        :param corpus_path: path to the corpus directory
        :param chunk_size: minimum chunk size per text in words
        :param language: language of the corpus
        :param cache_size: number of chunked texts to cache in memory
        """
        self.corpus_path = corpus_path
        self.chunk_size = chunk_size
        self.language = language
        self.cache_size = cache_size
    
    @abstractmethod
    def __iter__(self) -> CorpusParserIterator:
        """
        Iterator returning author pairs. This method is abstract and needs
        to be implemented by all concrete CorpusParsers.

        The returned iterator should be of type :class:`CorpusParser.CorpusParserIterator`

        :return: iterator object
        """
        pass
    
    def get_all_pairs(self) -> List[SamplePair]:
        """
        :return: list of all pairs in the current corpus
        """
        pairs = []
        for p in self:
            pairs.append(p)
        return pairs
