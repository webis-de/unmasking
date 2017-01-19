from nltk.tokenize import TreebankWordTokenizer

import nltk
import xxhash

from abc import ABC, abstractmethod
from typing import List
from enum import Enum, unique


_punctuation = [".", ",", ";", ":", "!", "?", "+", "-", "*", "/", "^", "°", "=", "~",  "$", "%",
                "(", ")", "[", "]", "{", "}", "<", ">",
                "`", "``", "'", "''", "--", "---"]


class SamplePair(object):
    """
    Pair of sample text sets.
    """
    
    @unique
    class Class(Enum):
        UNSPECIFIED = -1
        DIFFERENT_AUTHORS = 0
        SAME_AUTHOR = 1
    
    __sentence_tokenizers = {}
    __chunked_files = []
    
    def __init__(self, a: str, b: List[str], cls: Class, chunk_size: int,
                 language: str="english", cache_size: int=400):
        """
        Initialize pair of sample texts. Expects one main text ``a`` and one or more texts ``b``
        to compare with. Both ``a`` and ``b`` will be split into sets of sequential chunks
        according to ``chunk_size``. If ``chunk_size`` is smaller than the text length,
        only a single chunk will be produced. Chunks will always contain full sentences according
        to the NLTK Punkt tokenizer for the given ``language``.
        Texts in ``b`` will be chunked individually before adding them sequentially to the chunk list.
        
        :param a: text to verify
        :param b: list of other texts to compare with
        :param cls: class of the pair
        :param chunk_size: minimum chunk size per text in words
        :param language: language for sentence tokenization during chunk generation
        :param cache_size: how many chunked texts to cache in memory
        """
        self._cls = cls
        self._language = language
        self._cache_size = cache_size
        
        self._chunks_a = self._chunk_text(a, chunk_size)
        self._chunks_b = []
        for t in b:
            self._chunks_b.extend(self._chunk_text(t, chunk_size))
    
    @property
    def cls(self) -> Class:
        """Class (same author|different authors|unspecified)"""
        return self._cls
    
    @property
    def chunks_a(self) -> List[str]:
        """Chunks of first text (text to verify)"""
        return self._chunks_a
    
    @property
    def chunks_b(self) -> List[str]:
        """Chunks of texts to compare the first text (a) with"""
        return self._chunks_b
    
    @property
    def language(self) -> str:
        """Language of the sample texts a and b"""
        return self._language
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        text_hash = None
        if self._cache_size > 0:
            xxh = xxhash.xxh64()
            xxh.update(text)
            text_hash = xxh.digest()
            for f in self.__chunked_files:
                if f[0] == text_hash and f[1] == chunk_size:
                    return f[2]
        
        word_tokenizer = TreebankWordTokenizer()
        total_words = len([t for t in word_tokenizer.tokenize(text) if t not in _punctuation])
        num_chunks = total_words // chunk_size
        ideal_chunk_size = max(total_words // max(num_chunks, 1), chunk_size)
        
        if self.language not in self.__sentence_tokenizers:
            self.__sentence_tokenizers[self.language] = \
                nltk.data.load('tokenizers/punkt/{}.pickle'.format(self.language))
        
        sentences = self.__sentence_tokenizers[self.language].tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_chunk_size = 0
        for s in sentences:
            num_words = len([t for t in word_tokenizer.tokenize(s) if t not in _punctuation])
            current_chunk_size += num_words
            
            if current_chunk_size >= ideal_chunk_size:
                chunks.append(current_chunk)
                current_chunk = ""
                current_chunk_size = num_words
            
            if "" != current_chunk:
                current_chunk += " "
            current_chunk += s
        
        if 0 == len(chunks):
            # if minimum chunk size smaller than actual text, insert the only chunk we have
            chunks.append(current_chunk)
        else:
            # otherwise add left-over sentences to last chunk
            chunks[-1] += " " + current_chunk
            
            # combine last two chunks if the last chunk is too small
            if len(chunks) >= 2:
                last_chunk_len = len([t for t in word_tokenizer.tokenize(chunks[-1]) if t not in _punctuation])
                if last_chunk_len < chunk_size:
                    chunks[-2] += " " + chunks[-1]
                    del chunks[-1]
        
        if self._cache_size > 0:
            if len(self.__chunked_files) >= self._cache_size:
                del self.__chunked_files[0]
            self.__chunked_files.append((text_hash, chunk_size, chunks))
        
        return chunks
    

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

    def __init__(self, corpus_path: str, chunk_size: int, language: str="english", cache_size: int=400):
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
