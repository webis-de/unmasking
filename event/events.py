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

from event.interfaces import Event
from input.interfaces import SamplePair
from output.interfaces import Aggregator

from typing import List, Optional, Tuple


class ProgressEvent(Event):
    """
    Event for indicating progress of an operation with a fixed number of steps to be performed.
    """
    
    def __init__(self, group_id: str, serial: int, events_total: Optional[int] = None):
        """
        :param group_id: event group ID token
        :param serial: event serial number
        :param events_total: total number of events that will be sent in this event group
        """
        super().__init__(group_id, serial)
        
        if events_total is not None and events_total < 1:
            raise AttributeError("events_total must be greater than 0")

        self._events_total = events_total

    def clone(self) -> Event:
        return self.__class__(self.group_id, self.serial, self._events_total)

    @property
    def text(self) -> str:
        """Get user-readable textural representation of this event."""
        if self._events_total is None:
            return "Progress: {}".format(self.serial)

        return "Progress: {}/{} ({:.2f}%)".format(self.serial + 1, self.events_total, self.percent_done)

    @property
    def events_total(self) -> Optional[int]:
        """Get total number of events that will be sent in this event group."""
        return self._events_total

    @property
    def percent_done(self) -> Optional[float]:
        """Total progress in percent (None if total process is unknown)."""
        if self._events_total is None:
            return None

        return (float(self.serial) / self.events_total) * 100.0

    @property
    def finished(self) -> bool:
        """True if all operations have finished."""
        return self._events_total is not None and self.serial >= self._events_total


class PairChunkingProgressEvent(ProgressEvent):
    """
    Event for indicating pair chunking progress.
    """

    @property
    def text(self) -> str:
        """Get user-readable textural representation of this event."""
        if self.percent_done is not None:
            return "Chunking current pair: ({:.2f}%)".format(self.percent_done)

        return "Chunking current pair..."


class PairBuildingProgressEvent(ProgressEvent):
    """
    Event for status reports on pair generation.
    """

    def __init__(self, group_id: str, serial: int, pairs_total: Optional[int] = None, pair: SamplePair = None,
                 files_a: Optional[List[str]] = None, files_b: Optional[List[str]] = None):
        """
        :param group_id: event group ID token
        :param serial: event serial number
        :param pair: pair for which this event is emitted
        :param files_a: participating files for chunk set a
        :param files_b: participating files for chunk set b
        """
        super().__init__(group_id, serial, pairs_total)
        self._pair = pair
        self._files_a = [] if files_a is None else files_a
        self._files_b = [] if files_b is None else files_b

    def clone(self) -> Event:
        return self.__class__(self.group_id, self.serial, self._events_total,
                              self._pair, self._files_a, self._files_b)

    @property
    def pair(self):
        """Pair for which this event is emitted"""
        return self._pair

    @property
    def files(self) -> Tuple[List[str], List[str]]:
        """Lists of input files participating in this pair's generation of chunk sets a and b"""
        return self._files_a, self._files_b

    @files.setter
    def files(self, files: Tuple[List[str], List[str]]):
        """
        Set files participating in this pair's generation.

        :param files: participating files for chunk sets a and b as tuple of lists
        """
        self._files_a = files[0]
        self._files_b = files[1]

    @property
    def text(self) -> str:
        """Get user-readable textural representation of this event."""
        if self.events_total is None:
            return "Generated pair {}.".format(self.serial)

        return "Generated pair {} of {}.".format(self.serial + 1, self.events_total)


class ConfigurationFinishedEvent(Event):
    """
    Event fired after a job configuration has finished execution.
    """
    def __init__(self, group_id: str, serial: int, aggregators: List[Aggregator]):
        """
        :param group_id: event group ID token
        :param serial: event serial number
        :param aggregators: list of curve aggregators
        """
        super().__init__(group_id, serial)
        self._aggregators = aggregators

    @property
    def aggregators(self) -> List[Aggregator]:
        """Get aggregators associated with this event"""
        return self._aggregators

    def add_aggregator(self, aggregator: Aggregator):
        """
        Associate aggregator with this event.

        :param aggregator: aggregator to add
        """
        self._aggregators.append(aggregator)


class JobFinishedEvent(ConfigurationFinishedEvent):
    """
    Event fired when a job has finished execution.
    """
    pass


class UnmaskingTrainCurveEvent(Event):
    """
    Event for updating training curves of pairs during unmasking.
    """
    
    def __init__(self, group_id: str, serial: int, n: int = 0, pair: SamplePair = None, feature_set: type = None):
        """
        :param group_id: event group ID token
        :param serial: event serial number
        :param n: predicted final number of total values (should be set to the total number of unmasking iterations)
        :param pair: pair for which this curve is being calculated
        :param feature_set: feature set class used for generating this curve
        """
        super().__init__(group_id, serial)
        self._n = n
        self._values = []
        self._pair = pair
        self._feature_set = feature_set

    def clone(self) -> Event:
        clone = UnmaskingTrainCurveEvent(self.group_id, self.serial, self._n, self._pair, self._feature_set)
        clone._values = self._values
        return clone
    
    @property
    def pair(self):
        """Pair for which this curve is being calculated."""
        return self._pair
    
    @property
    def values(self) -> List[float]:
        """Accuracy values of curve (may be shorter than ``n``)."""
        return self._values
    
    @values.setter
    def values(self, points: List[float]):
        """Set curve points and update ``n`` if necessary."""
        self._values = points
        self._n = max(self._n, len(self._values))

    def value(self, point: float):
        """Add point to curve and update ``n`` if necessary."""
        self._values.append(point)
        self._n = max(self._n, len(self._values))
    
    @property
    def n(self) -> int:
        """
        Predicted number of data points (usually the number of unmasking iterations).
        This is not the actual number of curve points, which would be ``len(values)``.
        """
        return self._n
    
    @n.setter
    def n(self, n: int):
        """
        Update prediction of the number of final data points in the curve.
        This should be set to the number of unmasking iterations.
        """
        self._n = n
    
    @property
    def feature_set(self) -> type:
        """Feature set class used for generating this curve."""
        return self._feature_set
    
    @feature_set.setter
    def feature_set(self, fs: type):
        """Set feature set class used for generating this curve."""
        self._feature_set = fs
