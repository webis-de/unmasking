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

from classifier.interfaces import ChunkSampler
from input.interfaces import SamplePair

import random
from typing import Iterable, Tuple


class RandomOversampler(ChunkSampler):
    """
    ChunkSampler for generating chunk pairs by random oversampling.
    The larger set will be iterated in order and each item will be matched
    with a random item from the smaller set.
    
    If both sets a and b have the same amount of chunks, they will be matched 1:1 in order.
    """
    
    def generate_chunk_pairs(self, pair: SamplePair) -> Iterable[Tuple[str, str]]:
        len_a = len(pair.chunks_a)
        len_b = len(pair.chunks_b)
        
        if len_b > len_a:
            for b in pair.chunks_b:
                yield (pair.chunks_a[random.randint(0, len_a - 1)], b)
        elif len_a > len_b:
            for a in pair.chunks_a:
                yield (a, pair.chunks_b[random.randint(0, len_b - 1)])
        else:
            for i in range(0, len_a):
                yield (pair.chunks_a[i], pair.chunks_b[i])


class RandomUndersampler(ChunkSampler):
    """
    ChunkSampler for generating chunk pairs by random undersampling.
    The smaller set will be iterated in order and each item will be matched
    with a random item from the larger set.
    
    If both sets a and b have the same amount of chunks, they will be matched 1:1 in order.
    """
    
    def generate_chunk_pairs(self, pair: SamplePair) -> Iterable[Tuple[str, str]]:
        len_a = len(pair.chunks_a)
        len_b = len(pair.chunks_b)
        
        if len_b < len_a:
            for b in pair.chunks_b:
                yield (pair.chunks_a[random.randint(0, len_a - 1)], b)
        elif len_a < len_b:
            for a in pair.chunks_a:
                yield (a, pair.chunks_b[random.randint(0, len_b - 1)])
        else:
            for i in range(0, len_a):
                yield (pair.chunks_a[i], pair.chunks_b[i])


class UniqueRandomUndersampler(ChunkSampler):
    """
    ChunkSampler for generating chunk pairs by random undersampling.
    The smaller set will be iterated in order and each item will be matched
    with a random item from the larger set.
    
    Other than :class:`RandomUndersampler`, this sampler guarantees that no
    item will be picked twice from the smaller set.
    
    If both sets a and b have the same amount of chunks, they will be matched 1:1 in order.
    """
    
    def generate_chunk_pairs(self, pair: SamplePair) -> Iterable[Tuple[str, str]]:
        len_a = len(pair.chunks_a)
        len_b = len(pair.chunks_b)
        
        sampled_items = []
        
        if len_b < len_a:
            for b in pair.chunks_b:
                while True:
                    index = random.randint(0, len_a - 1)
                    if index not in sampled_items:
                        sampled_items.append(index)
                        break
                yield (pair.chunks_a[index], b)
        elif len_a < len_b:
            for a in pair.chunks_a:
                while True:
                    index = random.randint(0, len_b - 1)
                    if index not in sampled_items:
                        sampled_items.append(index)
                        break
                yield (a, pair.chunks_b[index])
        else:
            for i in range(0, len_a):
                yield (pair.chunks_a[i], pair.chunks_b[i])
