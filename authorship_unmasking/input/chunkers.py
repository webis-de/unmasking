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

from authorship_unmasking.conf.interfaces import instance_property, instance_list_property
from authorship_unmasking.input.interfaces import Chunker, Tokenizer
from authorship_unmasking.input.tokenizers import CharNgramTokenizer, DisjunctCharNgramTokenizer, WordTokenizer

import nltk

from functools import lru_cache
from random import randint
from typing import Any, Iterable, List


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
    def chunk(self, text: str) -> Iterable[Any]:
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
                 with_replacement: bool = True, delimiter: str = " "):
        """
        :param chunk_size: target chunk size
        :param num_chunks: number of chunks to generate
        :param tokenizer: tokenizer for generating tokens from which to draw (default: WordTokenizer())
        :param with_replacement: whether to draw with replacement or without and
                                 full refill once all words have been drawn
        :param delimiter: delimiter to put between tokens of the generated text
        """
        super().__init__(chunk_size)
        self._num_chunks = num_chunks
        self._with_replacement = with_replacement
        self._tokenizer = tokenizer
        self._tokenizer = WordTokenizer() if tokenizer is None else tokenizer
        self._delimiter = delimiter

    def chunk(self, text: str) -> Iterable[Any]:
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
                    chunk += self._delimiter
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
    def delimiter(self) -> str:
        """Delimiter to put between words of the generated text"""
        return self._delimiter

    @delimiter.setter
    def delimiter(self, delimiter: str):
        """Set delimiter to put between words of the generated text"""
        self._delimiter = delimiter

    @property
    def with_replacement(self) -> bool:
        """Whether to draw words with or without replacement"""
        return self._with_replacement

    @with_replacement.setter
    def with_replacement(self, with_replacement: bool):
        self._with_replacement = with_replacement


class RandomWordTokenChunker(RandomTokenChunker):
    """
    RandomTokenChunker which uses a WordTokenizer
    """

    def __init__(self, chunk_size: int = 600, num_chunks: int = 25,
                 with_replacement: bool = True, delimiter: str = " "):
        super().__init__(chunk_size, num_chunks, WordTokenizer(), with_replacement, delimiter)


class RandomCharNgramTokenChunker(RandomTokenChunker):
    """
    RandomTokenChunker which uses a CharNgramTokenizer
    """

    def __init__(self, chunk_size: int = 600, num_chunks: int = 25,
                 with_replacement: bool = True, delimiter: str = ""):
        super().__init__(chunk_size, num_chunks, CharNgramTokenizer(), with_replacement, delimiter)


class RandomDisjunctCharTokenNgramChunker(RandomTokenChunker):
    """
    RandomTokenChunker which uses a DisjunctCharNgramTokenizer
    """

    def __init__(self, chunk_size: int = 600, num_chunks: int = 25,
                 with_replacement: bool = True, delimiter: str = ""):
        super().__init__(chunk_size, num_chunks, DisjunctCharNgramTokenizer(), with_replacement, delimiter)


class MultiChunker(Chunker):
    """
    Chunker which generates multiple chunks from different sub chunkers.
    """
    def __init__(self):
        super().__init__()
        self._sub_chunkers = []

    def chunk(self, text: str) -> Iterable[Any]:
        """
        Return an iterator over chunks, where each chunk is a list of individual
        corresponding chunks generated by all sub chunkers. If the sub chunkers
        return different numbers of chunks, the shorter ones will be padded with None.

        :param text: input text
        :return: iterator over chunk lists
        """
        chunker_generators = [iter(c.chunk(text)) for c in self._sub_chunkers]

        while True:
            chunks = []
            non_null = False
            for chunker_gen in chunker_generators:
                try:
                    chunks.append(next(chunker_gen))
                    non_null = True
                except StopIteration:
                    chunks.append(None)
            if non_null:
                yield chunks
            else:
                break

    def add_sub_chunker(self, chunker: Chunker):
        """
        Add an individual sub chunker.

        :param chunker: chunker set to add
        """
        self._sub_chunkers.append(chunker)

    @instance_list_property(delegate_args=True)
    def sub_chunkers(self) -> List[Chunker]:
        """ Get sub chunkers. """
        return self._sub_chunkers

    @sub_chunkers.setter
    def sub_chunkers(self, chunker: List[Chunker]):
        """ Set sub chunkers. """
        self._sub_chunkers = chunker
