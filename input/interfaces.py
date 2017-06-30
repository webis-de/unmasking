from conf.interfaces import Configurable

from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import Iterable, AsyncGenerator, List
from uuid import UUID


class Tokenizer(ABC, Configurable):
    """
    Base class for tokenizers.
    
    Tokenizer properties with setters defined via @property.setter
    can be set at runtime via job configuration.
    """

    @abstractmethod
    def tokenize(self, text: str) -> Iterable[str]:
        """
        Tokenize given input text.

        :param text: input text
        :return: iterable of tokens generated from ``t``
        """
        pass


@unique
class SamplePairClass(Enum):
    """
    Base enumeration type for pairs. Members designating specific pair classes can be
    defined in sub-types of this enum type.
    """

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if other is None and self.value == -1:
            return True
        elif isinstance(other, self.__class__):
            return other.value == self.value
        elif isinstance(other, str):
            return other.upper() == self.__str__()
        elif isinstance(other, int):
            return other == self.value
        elif isinstance(other, bool):
            if self.value == -1:
                return False
            else:
                return bool(self.value) == other

    def __hash__(self):
        return self.__repr__().__hash__()


class SamplePair(ABC):
    """
    Pair of sample text sets.

    Events published by this class:

    * `onProgress`:          [type ProgressEvent]
                             fired to indicate pair chunking progress
    """

    SAMPLE_PAIR_NS = UUID("412bd9f0-4c61-4bb7-a7f2-c88be2f9555c")

    def __init__(self, cls: SamplePairClass, chunk_tokenizer: Tokenizer):
        """
        Initialize pair of sample texts. Expects a set of main texts ``a`` and one
        or more texts ``b`` to compare with.
        Texts in ``a`` and ``b`` will be chunked individually before adding them
        sequentially to the chunk list.

        :param a: list of texts to verify
        :param b: list of other texts to compare with
        :param cls: class of the pair
        :param chunk_tokenizer: chunk tokenizer
        """
        self._cls = cls
        self._chunk_tokenizer = chunk_tokenizer

    @abstractmethod
    def chunk(self, a: List[str], b: List[str]):
        """
        Create chunks from inputs.

        :param a: input texts one
        :param b: input texts two
        """
        pass

    @property
    @abstractmethod
    def cls(self) -> SamplePairClass:
        """Class (same author|different authors|unspecified)"""
        pass

    @property
    @abstractmethod
    def pair_id(self) -> str:
        """UUID string identifying a pair based on its set of texts."""
        pass

    @property
    @abstractmethod
    def chunks_a(self) -> List[str]:
        """Chunks of first text (text to verify)"""
        pass

    @property
    @abstractmethod
    def chunks_b(self) -> List[str]:
        """Chunks of texts to compare the first text (a) with"""
        pass


class CorpusParser(ABC, Configurable):
    """
    Base class for corpus parsers.
    """

    def __init__(self, chunk_tokenizer: Tokenizer, corpus_path: str = None):
        """
        :param corpus_path: path to the corpus directory
        :param chunk_tokenizer: chunk tokenizer
        """
        self._corpus_path = corpus_path
        self.chunk_tokenizer = chunk_tokenizer

    @property
    def corpus_path(self) -> str:
        """Get corpus path"""
        return self._corpus_path

    @corpus_path.setter
    def corpus_path(self, path: str):
        """Set corpus path"""
        self._corpus_path = path

    @abstractmethod
    async def __aiter__(self) -> AsyncGenerator[SamplePair, None]:
        """
        Asynchronous generator return parsed SamplePairs.
        """
        pass
