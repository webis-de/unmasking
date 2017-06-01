from input.interfaces import Tokenizer

import nltk

from typing import Iterable
from functools import lru_cache
from random import randint


class WordTokenizer(Tokenizer):
    """
    Word tokenizer based on NLTK's Treebank Word tokenizer which discards punctuation tokens.
    """
    
    punctuation = [".", ",", ";", ":", "!", "?", "+", "-", "*", "/", "^", "°", "=", "~", "$", "%",
                   "(", ")", "[", "]", "{", "}", "<", ">",
                   "`", "``", "'", "''", "--", "---"]
    
    @lru_cache(maxsize=10000)
    def tokenize(self, text: str) -> Iterable[str]:
        word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        return (t for t in word_tokenizer.tokenize(text) if t not in self.punctuation)


class CharNgramTokenizer(Tokenizer):
    """
    Character n-gram tokenizer.
    """
    
    def __init__(self, order: int = 3):
        """
        :param order: n-gram order (defaults to trigrams)
        """
        super().__init__()
        self._order = None
        self.order = order
    
    @property
    def order(self) -> int:
        """Get n-gram order"""
        return self._order
    
    @order.setter
    def order(self, order: int):
        """Set n-gram order"""
        if order < 1:
            raise ValueError("Order must be greater than zero")
        self._order = order
    
    def tokenize(self, text: str) -> Iterable[str]:
        for i in range(0, len(text) - self._order + 1):
            yield text[i:i + self._order]


class DisjunctCharNgramTokenizer(CharNgramTokenizer):
    """
    Tokenizer for producing disjunct character n-grams.
    If the input text length is not a multiple of the given n-gram order, the last n-gram will be discarded.
    E.g. "hello world" will become ["hel", "lo ", "wor"]
    """
    
    def tokenize(self, text: str) -> Iterable[str]:
        text_len = len(text)
        for i in range(0, text_len, self._order):
            if i + self._order > text_len:
                return
            
            yield text[i:i + self._order]


class PassthroughTokenizer(Tokenizer):
    """
    Tokenizer which returns the full input as a single token / chunk.
    Useful for chunking larger collections of individual short texts.
    """
    
    def tokenize(self, text: str) -> Iterable[str]:
        return [text]


class SentenceChunkTokenizer(Tokenizer):
    """
    Tokenizer to tokenize texts into chunks of ``chunk_size`` words without splitting sentences.
    If ``chunk_size`` is smaller than the text length, only a single chunk will be produced.
    Chunks will always contain full sentences according to the NLTK Punkt tokenizer for the given ``language``.
    
    Chunked texts can be cached in memory for faster repeated processing. By default,
    the cache size is limited to 400 texts.
    """
    
    def __init__(self, chunk_size: int = 500, language: str = "english"):
        """
        :param chunk_size: maximum chunk size
        :param language: language of the text
        """
        self._chunk_size = chunk_size
        self._language = language
    
    @property
    def chunk_size(self) -> int:
        """Get chunk size"""
        return self._chunk_size
    
    @chunk_size.setter
    def chunk_size(self, chunk_size: int):
        """Set chunk size"""
        self._chunk_size = chunk_size
    
    @property
    def language(self) -> str:
        """Get language"""
        return self._language
    
    @language.setter
    def language(self, language: str):
        """Set language"""
        self._language = language

    @lru_cache(maxsize=10000)
    def tokenize(self, text: str) -> Iterable[str]:
        word_tokenizer = WordTokenizer()
        total_words = len(list(word_tokenizer.tokenize(text)))
        num_chunks = total_words // self._chunk_size
        ideal_chunk_size = max(total_words // max(num_chunks, 1), self._chunk_size)

        sent_tokenizer = self._get_sent_tokenizer(self._language)
        sentences = sent_tokenizer.tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_chunk_size = 0
        for s in sentences:
            num_words = len(list(word_tokenizer.tokenize(s)))
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
                last_chunk_len = len(list(word_tokenizer.tokenize(chunks[-1])))
                if last_chunk_len < self._chunk_size:
                    chunks[-2] += " " + chunks[-1]
                    del chunks[-1]

        return chunks

    @lru_cache(maxsize=20)
    def _get_sent_tokenizer(self, lang: str):
        return nltk.data.load('tokenizers/punkt/{}.pickle'.format(lang))


class RandomWordChunkTokenizer(WordTokenizer):
    """
    Tokenizer to produce chunks of words by randomly drawing words from a given text.
    
    Drawing can either be done with replacement or without replacement and full refill
    once all words have been drawn from the pool.
    """

    def __init__(self, chunk_size: int = 600, num_chunks: int = 25, with_replacement: bool = True):
        """
        :param chunk_size: target chunk size
        :param num_chunks: number of chunks to generate
        :param with_replacement: whether to draw with replacement or without and
                                 full refill once all words have been drawn
        """
        self._chunk_size = chunk_size
        self._num_chunks = num_chunks
        self._with_replacement = with_replacement
    
    def tokenize(self, text: str) -> Iterable[str]:
        words = self._get_words(text)
        word_freq = nltk.FreqDist(words)
        num_words = len(words)
        drawn = {}
        num_drawn = 0
        
        for i in range(0, self._num_chunks):
            chunk = ""
            cur_chunk_size = 0
            
            while cur_chunk_size < self._chunk_size:
                word = words[randint(0, num_words - 1)]

                if not self._with_replacement:
                    if num_drawn < num_words and word in drawn and drawn[word] >= word_freq[word]:
                        continue
                    elif num_drawn >= num_words:
                        drawn = {}
                        num_drawn = 0
                    
                    drawn[word] = drawn.get(word, 0) + 1
                    num_drawn += 1
                
                if chunk != "":
                    chunk += " "
                chunk += word
                cur_chunk_size += 1
            
            yield chunk

    @lru_cache(maxsize=10000)
    def _get_words(self, text: str):
        return list(super().tokenize(text))
    
    @property
    def chunk_size(self) -> int:
        """Size of a chunk in words"""
        return self._chunk_size
    
    @chunk_size.setter
    def chunk_size(self, chunk_size: int):
        """Set chunk size in words"""
        self._chunk_size = chunk_size
    
    @property
    def num_chunks(self) -> int:
        """Number of chunks to produce"""
        return self._num_chunks
    
    @num_chunks.setter
    def num_chunks(self, num_chunks: int):
        """Set number of chunks to produce"""
        self._num_chunks = num_chunks
    
    @property
    def with_replacement(self) -> bool:
        """Whether to draw words with or without replacement"""
        return self._with_replacement
    
    @with_replacement.setter
    def with_replacement(self, with_replacement: bool):
        self._with_replacement = with_replacement
