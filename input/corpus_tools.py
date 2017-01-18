from nltk.tokenize.api import TokenizerI
from nltk import FreqDist

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
from enum import Enum, unique


class SamplePair(object):
    """
    Pair of sample text sets.
    """
    
    @unique
    class SampleClass(Enum):
        UNSPECIFIED = -1
        DIFFERENT_AUTHORS = 0
        SAME_AUTHOR = 1
    
    def __init__(self, text1: str, texts2: List[str], sample_class: SampleClass, tokenizer: TokenizerI):
        """
        :param text1: text to identify
        :param texts2: list of texts to compare with
        :param sample_class: class of the pair
        :param tokenizer: tokenizer for separating tokens
        """
        self.text1 = text1
        self.texts2 = texts2
        self.sample_class = sample_class
        self.tokenizer = tokenizer
        
        # cache variables
        self._tokens_text1 = None
        self._filter_func1 = None
        self._top_n_text1 = 0
        self._top_tokens_text1 = None
        self._tokens_texts2 = None
        self._filter_func2 = None
        self._top_n_texts2 = 0
        self._top_tokens_texts2 = None
    
    def get_tokens_text1(self, filter_func: Callable[[str], bool]=None) -> List[str]:
        """
        Get tokens from text1 according to ``tokenizer``.
        
        :param filter_func: optional filter function to remove tokens such as
                             punctuation from the token list
        :return: tokenized text
        """
        if self._filter_func1 == filter_func and self._tokens_text1 is not None:
            return self._tokens_text1
        
        tokens = self.tokenizer.tokenize(self.text1)
        
        if filter_func is not None:
            tokens = [t for t in tokens if filter_func(t)]
        
        self._tokens_text1 = tokens
        self._filter_func1 = filter_func
        return tokens

    def get_tokens_texts2(self, filter_func: Callable[[str], bool] = None) -> List[List[str]]:
        """
        Get tokens according to ``tokenizer`` from each text in the *texts2* set.

        :param filter_func: optional filter function to remove tokens such as
                             punctuation from the token lists
        :return: tokenized texts
        """
        if self._filter_func2 == filter_func and self._tokens_texts2 is not None:
            return self._tokens_texts2
        
        token_lists = []

        for text in self.texts2:
            tokens = self.tokenizer.tokenize(text)
            
            if filter_func is not None:
                tokens = [t for t in tokens if filter_func(t)]
            
            token_lists.append(tokens)
        
        self._tokens_texts2 = token_lists
        self._filter_func2 = filter_func
        return token_lists
    
    def get_top_tokens_text1(self, n: int, filter_func: Callable[[str], bool] = None) -> List[Tuple[str, int]]:
        """
        Get top n-most frequent tokens from text1.
        
        :param n: number of top tokens
        :param filter_func: optional filter function to remove tokens such as
                             punctuation from the token lists before counting tokens
        :return: the n most-frequent tokens or less if there are not enough tokens
        """
        if self._top_n_text1 == n and self._filter_func1 == filter_func and self._top_tokens_text1 is not None:
            return self._top_tokens_text1

        self._top_tokens_text1 = FreqDist(self.get_tokens_text1(filter_func)).most_common(n)
        return self._top_tokens_text1
    
    def get_top_tokens_texts2(self, n: int, filter_func: Callable[[str], bool] = None) -> List[Tuple[str, int]]:
        """
        Get average top n-most frequent tokens from the *texts2* set.
        
        :param n: number of top tokens
        :param filter_func: optional filter function to remove tokens such as
                             punctuation from the token lists before counting tokens
        :return: the n on average most-frequent tokens or less if there are not enough tokens
        """
        if self._top_n_texts2 == n and self._filter_func1 == filter_func and self._top_tokens_texts2 is not None:
            return self._top_tokens_texts2
        
        freq_dist = FreqDist()
        token_lists = self.get_tokens_texts2(filter_func)
        for tokens in token_lists:
            freq_dist.update(tokens)

        self._top_tokens_texts2 = freq_dist.most_common(n)
        return self._top_tokens_texts2

    
class ChunkedPair(object):
    """
    Chunked text representation of text pairs
    """
    
    def __init__(self, pair: SamplePair):
        """
        :param pair: text pair to be chunked
        """
        self.pair = pair
    
    @staticmethod
    def _get_chunks(tokens: List[str], chunk_size: int) -> List[List[str]]:
        chunks = []
        num_tokens = len(tokens)
        num_chunks = num_tokens // chunk_size
        actual_chunk_size = num_tokens // num_chunks
    
        for i in range(0, num_chunks):
            if i < num_chunks - 1:
                chunks.append(tokens[i * actual_chunk_size:i * actual_chunk_size + actual_chunk_size])
            else:
                chunks.append(tokens[i * actual_chunk_size:])
    
        return chunks
    
    def get_chunks_text1(self, chunk_size: int, filter_func: Callable[[str], bool]=None) -> List[List[str]]:
        """
        Get a chunked version of the first text from the given pair.
        Tokens ignored by the filter will not count to the minimum chunk size.

        :param chunk_size: minimum chunk size
        :param filter_func: optional filter function to remove tokens such as
                             punctuation from the token list
        :return: chunks of tokenized texts
        """
        tokens = self.pair.get_tokens_text1(filter_func)
        return self._get_chunks(tokens, chunk_size)

    def get_chunks_texts2(self, chunk_size: int, filter_func: Callable[[str], bool] = None) -> List[List[str]]:
        """
        Get a chunked version of the *texts2* set from the given pair.
        Tokens ignored by the filter will not count to the minimum chunk size.
        All texts of the *texts2* set will be flattened into a single list of chunks.

        :param chunk_size: minimum chunk size
        :param filter_func: optional filter function to remove tokens such as
                             punctuation from the token list
        :return: chunks of tokenized texts
        """
        chunks = []
        token_lists = self.pair.get_tokens_texts2(filter_func)
        for tokens in token_lists:
            chunks.extend(self._get_chunks(tokens, chunk_size))
        
        return chunks
    

class CorpusParser(ABC):
    """
    Base class for corpus parsers.
    """
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def __iter__(self):
        return self

    @abstractmethod
    def __next(self):
        pass
