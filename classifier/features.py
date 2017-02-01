from classifier.interfaces import FeatureSet
from classifier.chunking import SamplePair, ChunkSampler
from util.map import BoundedHashMap

import numpy
from nltk import FreqDist
from nltk.tokenize import TreebankWordTokenizer

from typing import List, Tuple, Iterable

_punctuation = [".", ",", ";", ":", "!", "?", "+", "-", "*", "/", "^", "°", "=", "~", "$", "%",
                "(", ")", "[", "]", "{", "}", "<", ">",
                "`", "``", "'", "''", "--", "---"]


class AvgWordFreqFeatureSet(FeatureSet):
    """
    Feature set using the average frequencies of the n most
    frequent words in both input chunk sets.
    
    :param cache_size: number of tokenized chunks to save in memory
    """
    
    # cache variables
    __tokenized_chunks = BoundedHashMap()
    
    def __init__(self, pair: SamplePair, sampler: ChunkSampler, cache_size: int = 2000):
        super().__init__(pair, sampler)
        
        self.__tokenized_chunks.maxlen = cache_size
        
        self._tokenizer = TreebankWordTokenizer()
        self._cache_size = cache_size
        
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
    
    def get_features_absolute(self, n: int) -> Iterable[numpy.ndarray]:
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
    
    def _tokenize(self, text) -> List[str]:
        if text in self.__tokenized_chunks:
            return self.__tokenized_chunks[text]
        
        tokens = [w for w in self._tokenizer.tokenize(text) if w not in _punctuation]
        
        # cache tokenized chunks
        self.__tokenized_chunks[text] = tokens
        
        return tokens
