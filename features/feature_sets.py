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

from features.interfaces import FeatureSet
from features.sampling import ChunkSampler
from input.interfaces import SamplePair
from input.interfaces import Tokenizer
from input.tokenizers import WordTokenizer, CharNgramTokenizer, DisjunctCharNgramTokenizer
from util.util import lru_cache

import numpy
from nltk import FreqDist

from typing import List, Iterable


class CachedAvgTokenCountFeatureSet(FeatureSet):
    """
    Generic feature set which uses the average frequency counts per chunk of the
    tokens generated by a specified tokenizer and caches them in memory.
    By default, the cache size is limited to 2000 chunks.
    """
    def __init__(self, pair: SamplePair, sampler: ChunkSampler, chunk_tokenizer: Tokenizer):
        """
        :param pair: pair of chunked texts
        :param sampler: :class:`ChunkSampler` for sampling chunks from ``pair``
        :param chunk_tokenizer: tokenizer for tokenizing chunks
        """
        super().__init__(pair, sampler)

        self._chunk_tokenizer = chunk_tokenizer
        self._is_prepared = False

        self.__freq_a = None
        self.__freq_b = None
        self._chunks  = []

    def _prepare(self):
        if self._is_prepared:
            return

        freq_dist_a = FreqDist()
        for a in self._pair.chunks_a:
            freq_dist_a.update(self._tokenize(a))

        freq_dist_b = FreqDist()
        for b in self._pair.chunks_b:
            freq_dist_b.update(self._tokenize(b))

        self._avg_freq_dist = FreqDist()
        n_a = freq_dist_a.N()
        n_b = freq_dist_b.N()
        for a in freq_dist_a:
            self._avg_freq_dist[a] = (freq_dist_a[a] / n_a + freq_dist_b[a] / n_b) / 2.0
        for b in freq_dist_b:
            if self._avg_freq_dist[b] != 0.0:
                continue
            self._avg_freq_dist[b] = (freq_dist_a[b] / n_a + freq_dist_b[b] / n_b) / 2.0

        self._chunks = self._sampler.generate_chunk_pairs(self._pair)

        self.__freq_a = None
        self.__freq_b = None

        self._is_prepared = True

    def get_features_absolute(self, n: int) -> Iterable[numpy.ndarray]:
        self._prepare()

        top_n_words = numpy.array([w for (w, f) in self._avg_freq_dist.most_common(n)])
        num_top_words = len(top_n_words)
        for c in self._chunks:
            vec = numpy.zeros(2 * n)

            self.__freq_a = FreqDist(self._tokenize(c[0]))

            for i in range(0, n):
                if i >= num_top_words:
                    break
                vec[i] = self.__freq_a[top_n_words[i]]

            self.__freq_b = FreqDist(self._tokenize(c[1]))

            for i in range(n, 2 * n):
                if i >= num_top_words + n:
                    break
                vec[i] = self.__freq_b[top_n_words[i - n]]

            yield vec

    def get_features_relative(self, n: int) -> Iterable[numpy.ndarray]:
        features = self.get_features_absolute(n)
        for vec in features:
            n_a = self.__freq_a.N()
            for i in range(0, n):
                vec[i] /= n_a
            n_b = self.__freq_b.N()
            for i in range(n, 2 * n):
                vec[i] /= n_b
        
            yield vec

    @lru_cache(maxsize=10000)
    def _tokenize(self, text) -> List[str]:
        return list(self._chunk_tokenizer.tokenize(text))


class AvgWordFreqFeatureSet(CachedAvgTokenCountFeatureSet):
    """
    Feature set using the average frequencies of the n most
    frequent words in both input chunk sets.
    """
    
    def __init__(self, pair: SamplePair, sampler: ChunkSampler):
        super().__init__(pair, sampler, WordTokenizer())


class AvgCharNgramFreqFeatureSet(CachedAvgTokenCountFeatureSet):
    """
    Feature set using the average frequencies of the k most
    frequent character n-grams in both input chunk sets.

    Default n-gram order is 3.
    """

    def __init__(self, pair: SamplePair, sampler: ChunkSampler):
        self.__tokenizer = CharNgramTokenizer(3)
        super().__init__(pair, sampler, self.__tokenizer)

    @property
    def order(self) -> int:
        """ Get n-gram order. """
        return self.__tokenizer.order

    @order.setter
    def order(self, ngram_order: int):
        """ Set n-gram order. """
        self.__tokenizer.order = ngram_order


class AvgDisjunctCharNgramFreqFeatureSet(CachedAvgTokenCountFeatureSet):
    """
    Feature set using the average frequencies of the k most
    frequent character n-grams in both input chunk sets.

    Default n-gram order is 3.
    """

    def __init__(self, pair: SamplePair, sampler: ChunkSampler):
        self.__tokenizer = DisjunctCharNgramTokenizer(3)
        super().__init__(pair, sampler, self.__tokenizer)

    @property
    def order(self) -> int:
        """ Get n-gram order. """
        return self.__tokenizer.order

    @order.setter
    def order(self, ngram_order: int):
        """ Set n-gram order. """
        self.__tokenizer.order = ngram_order