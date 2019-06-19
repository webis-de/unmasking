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

from authorship_unmasking.job.interfaces import ConfigurationExpander

from itertools import product
from typing import Iterable, Tuple


class ZipExpander(ConfigurationExpander):
    """
    Generate n configurations for m vectors containing n values each. Vectors must have the same length.
    If configurations vectors are of different lengths, values in longer vectors are ignored.

    For example, three vectors ("a", "b"), ("c", "d"), ("e", "f") will be expanded to two configurations
    ("a", "c", "e") and ("b", "d", "f").

    Generates output equivalent to Python's zip() function.
    """
    def expand(self, configuration_vectors: Iterable[Tuple]) -> Iterable[Tuple]:
        return zip(*configuration_vectors)


class ProductExpander(ConfigurationExpander):
    """
    Expand configuration to the Cartesian product of vector values with all n^m (or n1*n2*...*nn for
    vectors of different lengths) possible combinations.

    For example, three vectors ("a", "b"), ("c", "d"), ("e", "f") will be expanded to eight configurations
    ("a", "c", "e"), ("a", "c", "f"),
    ("a", "d", "e"), ("a", "d", "f"),
    ("b", "c", "e"), ("b", "c", "f"),
    ("b", "d", "e"), ("b", "d", "f").

    Generates output equivalent to Python's itertools.product() function.

    WARNING: building the Cartesian product is linear to the number of dimensions of a single
    vector, but exponential to the overall number of vectors. Make sure you have enough memory
    and use as few different vectors as possible.
    """
    def expand(self, configuration_vectors: Iterable[Tuple]) -> Iterable[Tuple]:
        return product(*configuration_vectors)
