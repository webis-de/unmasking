from classifier.interfaces import FeatureSet
from classifier.chunking import SamplePair, ChunkSampler

from nltk import FreqDist
from nltk.tokenize import TreebankWordTokenizer
import xxhash

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
    
    __tokenized_chunks = {}
    __tokenized_chunks_hashes = []

    def __init__(self, pair: SamplePair, sampler: ChunkSampler, cache_size: int=2000):
        super().__init__(pair, sampler)
        
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

    def get_features_absolute(self, n: int) -> Iterable[Tuple[List[float], SamplePair.Class]]:
        top_n_words = [w for (w, f) in self._avg_freq_dist.most_common(n)]
        for c in self._chunks:
            a = FreqDist(self._tokenize(c[0]))
            a_n = a.N()
            b = FreqDist(self._tokenize(c[1]))
            b_n = b.N()
            vec = [0.0] * n
            for i in range(0, n):
                if i >= len(top_n_words):
                    break
                word = top_n_words[i]
                vec[i] = (a[word] / a_n + b[word] / b_n) / 2.0
            
            yield (vec, self._pair.cls)
    
    def get_features_relative(self, n: int) -> Iterable[Tuple[List[float], SamplePair.Class]]:
        raise NotImplementedError
    
    def _tokenize(self, text) -> List[str]:
        chunk_hash = None
        if self._cache_size > 0:
            xxh = xxhash.xxh64()
            xxh.update(text)
            chunk_hash = xxh.digest()
            if chunk_hash in self.__tokenized_chunks:
                return self.__tokenized_chunks[chunk_hash]
        
        tokens = [w for w in self._tokenizer.tokenize(text) if w not in _punctuation]
        
        if self._cache_size > 0:
            if len(self.__tokenized_chunks_hashes) >= self._cache_size:
                # delete oldest cache entry by regenerating the dict (faster than del)
                self.__tokenized_chunks = {k: v for (k, v) in
                                           self.__tokenized_chunks.items() if k != self.__tokenized_chunks_hashes[0]}
                del self.__tokenized_chunks_hashes[0]
            self.__tokenized_chunks[chunk_hash] = tokens
            self.__tokenized_chunks_hashes.append(chunk_hash)
        
        return tokens
