from input.interfaces import Tokenizer
from util import BoundedHashMap

import nltk

from typing import Iterable

from util.cache import CacheMixin


class WordTokenizer(Tokenizer):
    """
    Word tokenizer based on NLTK's Treebank Punkt tokenizer which discards punctuation tokens.
    """
    
    def __init__(self, language: str = "english"):
        """
        :param language: language to use for the tokenizer
        """
        self._language = language
        self._punctuation = [".", ",", ";", ":", "!", "?", "+", "-", "*", "/", "^", "°", "=", "~", "$", "%",
                             "(", ")", "[", "]", "{", "}", "<", ">",
                             "`", "``", "'", "''", "--", "---"]
    
    def tokenize(self, t: str) -> Iterable[str]:
        word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        return (t for t in word_tokenizer.tokenize(t) if t not in self._punctuation)


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

    def tokenize(self, t: str) -> Iterable[str]:
        cached_chunks = self.get_cache_item(self._chunks_handle, t)
        if cached_chunks is not None:
            return cached_chunks
        
        word_tokenizer = WordTokenizer(self._language)
        total_words = len(list(word_tokenizer.tokenize(t)))
        num_chunks = total_words // self._chunk_size
        ideal_chunk_size = max(total_words // max(num_chunks, 1), self._chunk_size)
    
        sent_tokenizer = self.get_cache_item(self._tokenizers_handle, self._language)
        if sent_tokenizer is None:
            sent_tokenizer = nltk.data.load('tokenizers/punkt/{}.pickle'.format(self._language))
            self.set_cache_item(self._chunks_handle, self._language, sent_tokenizer)
    
        sentences = sent_tokenizer.tokenize(t)
        
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
        self.set_cache_item(self._chunks_handle, t, chunks)
        
        return chunks
