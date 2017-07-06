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

from conf.interfaces import Configurable
from input.interfaces import SamplePairClass

from abc import ABC, abstractmethod
from time import time
from typing import Any, Dict, List, Tuple


class Output(Configurable, ABC):
    """
    Base class for output handlers
    """
    
    @abstractmethod
    def save(self, output_dir: str):
        """
        Save object state to file in a given output directory

        :param output_dir: output directory
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset output and clear all variable data"""
        pass
    
    def _get_output_filename_base(self) -> str:
        """
        Generate a base filename (without extension) containing a timestamp which
        can safely be used for writing output files.
        
        :return: filename base
        """
        return self.__class__.__name__ + "." + str(int(time()))


class Aggregator(Output, ABC):
    """
    Base class for unmasking curve aggregation.
    This can be used for building ensembles of multiple runs.
    """

    @abstractmethod
    def add_curve(self, identifier: str, cls: SamplePairClass, values: List[float]):
        """
        Add curve to aggregation for given class.

        :param identifier: a common identifier for this and further instances of this curve
        :param cls: class of the pair
        :param values: curve points
        """
        pass

    @abstractmethod
    def get_aggregated_curves(self) -> Dict[str, Any]:
        """
        Return aggregated curves for each sample pair.

        :return: dictionary containing aggregation keys and curve data (may be another dictionary)
        """
        pass
