from event import EventBroadcaster, ProgressEvent
from util.map import BoundedHashMap

import nltk
from nltk.tokenize import TreebankWordTokenizer

import random
from abc import ABC, abstractmethod
from typing import List, Iterable, Tuple
from enum import Enum, unique

_punctuation = [".", ",", ";", ":", "!", "?", "+", "-", "*", "/", "^", "°", "=", "~", "$", "%",
                "(", ")", "[", "]", "{", "}", "<", ">",
                "`", "``", "'", "''", "--", "---"]


class SamplePair:
    """
    Pair of sample text sets.
    
    Events published by this class:
    
    * `progress`: progress of chunk generation
    """
    
    @unique
    class Class(Enum):
        UNSPECIFIED = -1
        DIFFERENT_AUTHORS = 0
        SAME_AUTHOR = 1

        def __repr__(self):
            return self.name
        
        def __str__(self):
            return self.__repr__()
    
    # cache variables
    __sentence_tokenizers = {}
    __chunked_files = BoundedHashMap()
    
    def __init__(self, a: str, b: List[str], cls: Class, chunk_size: int,
                 language: str = "english", cache_size: int = 400):
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
        self.__chunked_files.maxlen = cache_size
        
        self._cls = cls
        self._language = language
        self._cache_size = cache_size
        
        self._progress_event = ProgressEvent("progress", len(b) + 1)
        EventBroadcaster.publish(self._progress_event, self.__class__)
        
        self._chunks_a = self._chunk_text(a, chunk_size)
        self._progress_event.increment()
        EventBroadcaster.publish(self._progress_event, self.__class__)
        
        self._chunks_b = []
        for t in b:
            self._chunks_b.extend(self._chunk_text(t, chunk_size))
            self._progress_event.increment()
            EventBroadcaster.publish(self._progress_event, self.__class__)
    
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
        if text in self.__chunked_files:
            return self.__chunked_files[text]
        
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
        
        # cache chunked files
        self.__chunked_files[text] = chunks
        
        return chunks


class ChunkSampler(ABC):
    """
    Base class for chunk samplers used for generating pairs of chunks from :class:`SamplePair`s.
    """
    
    @abstractmethod
    def generate_chunk_pairs(self, pair: SamplePair) -> Iterable[Tuple[str, str]]:
        """
        Generate pairs of chunks from the given :class:`SamplePair.

        :param pair: text pair to create chunk pairs from
        :return: generator or other iterable producing the chunk pairs
        """
        pass


class RandomOversampler(ChunkSampler):
    """
    ChunkSampler for generating chunk pairs by random oversampling.
    The larger set will be iterated in order and each item will be matched
    with a random item from the smaller set.
    
    If both sets a and b have the same amount of chunks, they will be matched 1:1 in order.
    """
    
    def generate_chunk_pairs(self, pair: SamplePair) -> Iterable[Tuple[str, str]]:
        len_a = len(pair.chunks_a)
        len_b = len(pair.chunks_b)
        
        if len_b > len_a:
            for b in pair.chunks_b:
                yield (pair.chunks_a[random.randint(0, len_a - 1)], b)
        elif len_a > len_b:
            for a in pair.chunks_a:
                yield (a, pair.chunks_b[random.randint(0, len_b - 1)])
        else:
            for i in range(0, len_a):
                yield (pair.chunks_a[i], pair.chunks_b[i])


class RandomUndersampler(ChunkSampler):
    """
    ChunkSampler for generating chunk pairs by random undersampling.
    The smaller set will be iterated in order and each item will be matched
    with a random item from the larger set.
    
    If both sets a and b have the same amount of chunks, they will be matched 1:1 in order.
    """
    
    def generate_chunk_pairs(self, pair: SamplePair) -> Iterable[Tuple[str, str]]:
        len_a = len(pair.chunks_a)
        len_b = len(pair.chunks_b)
        
        if len_b < len_a:
            for b in pair.chunks_b:
                yield (pair.chunks_a[random.randint(0, len_a - 1)], b)
        elif len_a < len_b:
            for a in pair.chunks_a:
                yield (a, pair.chunks_b[random.randint(0, len_b - 1)])
        else:
            for i in range(0, len_a):
                yield (pair.chunks_a[i], pair.chunks_b[i])


class UniqueRandomUndersampler(ChunkSampler):
    """
    ChunkSampler for generating chunk pairs by random undersampling.
    The smaller set will be iterated in order and each item will be matched
    with a random item from the larger set.
    
    Other than :class:`RandomUndersampler`, this sampler guarantees that no
    item will be picked twice from the smaller set.
    
    If both sets a and b have the same amount of chunks, they will be matched 1:1 in order.
    """
    
    def generate_chunk_pairs(self, pair: SamplePair) -> Iterable[Tuple[str, str]]:
        len_a = len(pair.chunks_a)
        len_b = len(pair.chunks_b)
        
        sampled_items = []
        
        if len_b < len_a:
            for b in pair.chunks_b:
                while True:
                    index = random.randint(0, len_a - 1)
                    if index not in sampled_items:
                        sampled_items.append(index)
                        break
                yield (pair.chunks_a[index], b)
        elif len_a < len_b:
            for a in pair.chunks_a:
                while True:
                    index = random.randint(0, len_b - 1)
                    if index not in sampled_items:
                        sampled_items.append(index)
                        break
                yield (a, pair.chunks_b[index])
        else:
            for i in range(0, len_a):
                yield (pair.chunks_a[i], pair.chunks_b[i])
