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

from unmasking.interfaces import UnmaskingStrategy

import asyncio
import numpy


class FeatureRemoval(UnmaskingStrategy):
    """
    Classic feature removal as suggested by Koppel et al.
    """

    def __init__(self, num_eliminate: int = 10):
        """
        :param num_eliminate: number of features to eliminate per iteration.
        """
        super().__init__()
        self._num_eliminate = num_eliminate
    
    @property
    def eliminate(self) -> int:
        """Get number of eliminations per round"""
        return self._num_eliminate
    
    @eliminate.setter
    def eliminate(self, eliminate: int):
        """Set number of eliminations per round"""
        self._num_eliminate = eliminate

    async def transform(self, data: numpy.ndarray, coef: numpy.ndarray) -> numpy.ndarray:
        coef = numpy.absolute(coef)
        
        for i in range(0, self._num_eliminate):
            if data.shape[1] == 0:
                return data
            
            index = numpy.argmax(coef)
            coef  = numpy.delete(coef, index)
            data  = numpy.delete(data, index, 1)

            await asyncio.sleep(0)
        
        return data
