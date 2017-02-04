from input.interfaces import Tokenizer
from util.cache import CacheMixin

import nltk

from typing import Iterable


class WordTokenizer(Tokenizer):
    """
    Word tokenizer based on NLTK's Treebank Word tokenizer which discards punctuation tokens.
    """
    
    punctuation = [".", ",", ";", ":", "!", "?", "+", "-", "*", "/", "^", "°", "=", "~", "$", "%",
                   "(", ")", "[", "]", "{", "}", "<", ">",
                   "`", "``", "'", "''", "--", "---"]
    
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


class PassthroughTokenizer(Tokenizer):
    """
    Tokenizer which returns the full input as a single token / chunk.
    Useful for chunking larger collections of individual short texts.
    """
    
    def tokenize(self, text: str) -> Iterable[str]:
        return [text]


class SentenceChunkTokenizer(Tokenizer, CacheMixin):
    """
    Tokenizer to tokenize texts into chunks of ``chunk_size`` words without splitting sentences.
    If ``chunk_size`` is smaller than the text length, only a single chunk will be produced.
    Chunks will always contain full sentences according to the NLTK Punkt tokenizer for the given ``language``.
    
    Chunked texts can be cached in memory for faster repeated processing. By default,
    the cache size is limited to 400 texts.
    """
    
    def __init__(self, chunk_size: int, language: str = "english"):
        """
        :param chunk_size: maximum chunk size
        :param language: language of the text
        """
        self._chunks_handle = self.resolve_cache_alias(self.__class__.__name__ + "_chunked_texts")
        if -1 == self._chunks_handle:
            self._chunks_handle = self.init_cache(2000)
            self.set_cache_alias(self._chunks_handle, self.__class__.__name__ + "_chunked_texts")
            
        self._tokenizers_handle = self.resolve_cache_alias(self.__class__.__name__ + "_sent_tokenizers")
        if -1 == self._tokenizers_handle:
            self._tokenizers_handle = self.init_cache(0)
            self.set_cache_alias(self._tokenizers_handle, self.__class__.__name__ + "_sent_tokenizers")
        
        self._chunk_size = chunk_size
        self._language = language

    def tokenize(self, text: str) -> Iterable[str]:
        cached_chunks = self.get_cache_item(self._chunks_handle, text)
        if cached_chunks is not None:
            return cached_chunks
        
        word_tokenizer = WordTokenizer()
        total_words = len(list(word_tokenizer.tokenize(text)))
        num_chunks = total_words // self._chunk_size
        ideal_chunk_size = max(total_words // max(num_chunks, 1), self._chunk_size)
    
        sent_tokenizer = self.get_cache_item(self._tokenizers_handle, self._language)
        if sent_tokenizer is None:
            sent_tokenizer = nltk.data.load('tokenizers/punkt/{}.pickle'.format(self._language))
            self.set_cache_item(self._chunks_handle, self._language, sent_tokenizer)
    
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
    
        # cache chunked text
        self.set_cache_item(self._chunks_handle, text, chunks)
        
        return chunks
