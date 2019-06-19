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

from authorship_unmasking.conf.interfaces import Configurable
from authorship_unmasking.input.interfaces import SamplePairClass

from abc import ABCMeta, abstractmethod
from time import time
from typing import Any, Dict, List, Optional


class Output(Configurable, metaclass=ABCMeta):
    """
    Base class for output handlers
    """
    
    @abstractmethod
    async def save(self, output_dir: str, file_name: Optional[str] = None):
        """
        Save object state to file in a given output directory

        :param output_dir: output directory
        :param file_name: file name inside the output directory (None for auto-generated name)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset output and clear all variable data"""
        pass

    def _generate_output_basename(self) -> str:
        """
        Generate a base filename (without extension) containing a timestamp which
        can safely be used for writing output files.
        
        :return: filename base
        """
        return self.__class__.__name__ + "." + str(int(time()))


class Aggregator(Output, metaclass=ABCMeta):
    """
    Base class for unmasking curve aggregation.
    This can be used for building ensembles of multiple runs.
    """

    def __init__(self, meta_data: Dict[str, Any] = None):
        self._initial_meta_data = meta_data if meta_data is not None else {}
        self._meta_data = self._initial_meta_data

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

    @abstractmethod
    def get_aggregated_output(self) -> Output:
        """
        :return: configured output instance containing aggregated results
        """
        pass

    def reset(self):
        self.__init__(self._initial_meta_data)

    @property
    def meta_data(self) -> Dict[str, Any]:
        """Get experiment meta data"""
        return self._meta_data

    @meta_data.setter
    def meta_data(self, meta_data: Dict[str, Any]):
        """Add experiment meta data"""
        self._meta_data.update(meta_data)
