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

from input.interfaces import Tokenizer
from util.util import lru_cache

import nltk

from typing import Iterable


class WordTokenizer(Tokenizer):
    """
    Word tokenizer based on NLTK's Treebank Word tokenizer which discards punctuation tokens.
    """
    
    punctuation = [".", ",", ";", ":", "!", "?", "+", "-", "*", "/", "^", "°", "=", "~", "$", "%",
                   "(", ")", "[", "]", "{", "}", "<", ">",
                   "`", "``", "'", "''", "--", "---"]

    def __init__(self):
        self._tokenizer = nltk.tokenize.TreebankWordTokenizer()

    @lru_cache(maxsize=700)
    def tokenize(self, text: str) -> Iterable[str]:
        return [t for t in self._tokenizer.tokenize(text) if t not in self.punctuation]


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
        yield text
