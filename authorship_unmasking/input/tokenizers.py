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
