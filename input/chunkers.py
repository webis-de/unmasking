from conf.interfaces import instance_property
from input.interfaces import Chunker, Tokenizer
from input.tokenizers import WordTokenizer

import nltk

from functools import lru_cache
from random import randint
from typing import Iterable


class SentenceChunker(Chunker):
    """
    Chunk input texts into pieces of ``chunk_size`` words without splitting sentences.
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
        super().__init__(chunk_size)
        self._language = language
        self._word_tokenizer = WordTokenizer()

    @property
    def language(self) -> str:
        """Get language"""
        return self._language

    @language.setter
    def language(self, language: str):
        """Set language"""
        self._language = language

    @lru_cache(maxsize=500)
    def chunk(self, text: str) -> Iterable[str]:
        word_tokens = self._word_tokenizer.tokenize(text)
        assert type(word_tokens) is list
        # noinspection PyTypeChecker
        total_words = len(word_tokens)
        num_chunks = total_words // self._chunk_size
        ideal_chunk_size = max(total_words // max(num_chunks, 1), self._chunk_size)

        sent_tokenizer = self._get_sent_tokenizer(self._language)
        sentences = sent_tokenizer.tokenize(text)

        chunks = []
        current_chunk = ""
        current_chunk_size = 0
        for s in sentences:
            # noinspection PyTypeChecker
            num_words = len(self._word_tokenizer.tokenize(s))
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
                # noinspection PyTypeChecker
                last_chunk_len = len(self._word_tokenizer.tokenize(chunks[-1]))
                if last_chunk_len < self._chunk_size:
                    chunks[-2] += " " + chunks[-1]
                    del chunks[-1]

        return chunks

    @lru_cache(maxsize=20)
    def _get_sent_tokenizer(self, lang: str):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading nltk punkt tokenizer. This has to be done only once.")
            nltk.download('punkt')

        return nltk.data.load('tokenizers/punkt/{}.pickle'.format(lang))


class RandomTokenChunker(Chunker):
    """
    Chunker to produce chunks of words by randomly drawing words from a given text.

    Drawing can either be done with replacement or without replacement and full refill
    once all words have been drawn from the pool.
    """

    def __init__(self, chunk_size: int = 600, num_chunks: int = 25, tokenizer: Tokenizer = None,
                 with_replacement: bool = True):
        """
        :param chunk_size: target chunk size
        :param num_chunks: number of chunks to generate
        :param tokenizer: tokenizer for generating tokens from which to draw (default: WordTokenizer())
        :param with_replacement: whether to draw with replacement or without and
                                 full refill once all words have been drawn
        """
        super().__init__(chunk_size)
        self._num_chunks = num_chunks
        self._with_replacement = with_replacement
        self._tokenizer = tokenizer
        self._tokenizer = WordTokenizer() if tokenizer is None else tokenizer

    def chunk(self, text: str) -> Iterable[str]:
        tokens = self._tokenizer.tokenize(text)
        if type(tokens) is not list:
            tokens = list(tokens)
        word_freq = nltk.FreqDist(tokens)
        num_words = len(tokens)
        drawn = {}
        num_drawn = 0

        for i in range(0, self._num_chunks):
            chunk = ""
            cur_chunk_size = 0

            while cur_chunk_size < self._chunk_size:
                # noinspection PyUnresolvedReferences
                word = tokens[randint(0, num_words - 1)]

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

    @property
    def num_chunks(self) -> int:
        """Number of chunks to produce"""
        return self._num_chunks

    @num_chunks.setter
    def num_chunks(self, num_chunks: int):
        """Set number of chunks to produce"""
        self._num_chunks = num_chunks

    @instance_property
    def tokenizer(self) -> Tokenizer:
        """Tokenizer to generate chunks from"""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Tokenizer):
        """Set tokenizer to generate chunks from"""
        self._tokenizer = tokenizer

    @property
    def with_replacement(self) -> bool:
        """Whether to draw words with or without replacement"""
        return self._with_replacement

    @with_replacement.setter
    def with_replacement(self, with_replacement: bool):
        self._with_replacement = with_replacement
