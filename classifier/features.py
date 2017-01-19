from classifier.interfaces import FeatureSet
from classifier.chunking import SamplePair, ChunkSampler

from nltk import FreqDist
from nltk.tokenize import TreebankWordTokenizer

from typing import List, Iterable


_punctuation = [".", ",", ";", ":", "!", "?", "+", "-", "*", "/", "^", "°", "=", "~", "$", "%",
                "(", ")", "[", "]", "{", "}", "<", ">",
                "`", "``", "'", "''", "--", "---"]


class WordFreqFeatureSet(FeatureSet):
    """
    Feature set using the frequencies of the n most frequent words in both texts.
    """

    def __init__(self, pair: SamplePair, sampler: ChunkSampler):
        super().__init__(pair, sampler)
        
        self._tokenizer = TreebankWordTokenizer()
        
        self._freq_dist_a = FreqDist()
        for a in self._pair.chunks_a:
            self._freq_dist_a.update((w for w in self._tokenizer.tokenize(a) if w not in _punctuation))

        self._freq_dist_b = FreqDist()
        for b in self._pair.chunks_b:
            self._freq_dist_b.update((w for w in self._tokenizer.tokenize(b) if w not in _punctuation))
        
        self._chunks = self._sampler.generate_chunk_pairs(self._pair)

    def get_features_absolute(self, n: int) -> Iterable[List[float]]:
        raise NotImplementedError
    
    def get_features_relative(self, n: int) -> Iterable[List[float]]:
        raise NotImplementedError
