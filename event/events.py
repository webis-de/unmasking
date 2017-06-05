from event.interfaces import Event

from typing import List, Optional, Tuple


class ProgressEvent(Event):
    """
    Event for indicating progress of an operation with a fixed number of steps to be performed.
    """
    
    def __init__(self, group_id: str, serial: int, events_total: int = 1):
        """
        :param group_id: event group ID token
        :param serial: event serial number
        :param events_total: total number of events that will be sent in this event group
        """
        super().__init__(group_id, serial)
        
        if events_total < 1:
            raise AttributeError("events_total must be greater than 0")

        self._events_total = events_total

    def clone(self) -> Event:
        return ProgressEvent(self.group_id, self.serial, self._events_total)
    
    @property
    def events_total(self) -> int:
        """Get total number of events that will be sent in this event group."""
        return self._events_total
    
    @property
    def percent_done(self) -> float:
        """Total progress in percent."""
        return (float(self.serial) / self.events_total) * 100.0
    
    @property
    def finished(self) -> bool:
        """True if all operations have finished."""
        return self.serial >= self._events_total


class UnmaskingTrainCurveEvent(Event):
    """
    Event for updating training curves of pairs during unmasking.
    """
    
    def __init__(self, group_id: str, serial: int, n: int = 0, pair: "SamplePair" = None, feature_set: type = None):
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


class PairGenerationEvent(Event):
    """
    Event for status reports on pair generation.
    """
    
    def __init__(self, group_id: str, serial: int, pair: "SamplePair" = None,
                 files_a: Optional[List[str]] = None, files_b: Optional[List[str]] = None):
        """
        :param group_id: event group ID token
        :param serial: event serial number
        :param pair: pair for which this event is emitted
        :param files_a: participating files for chunk set a
        :param files_b: participating files for chunk set b
        """
        super().__init__(group_id, serial)
        self._pair = pair
        self._files_a = [] if files_a is None else files_a
        self._files_b = [] if files_b is None else files_b

    def clone(self) -> Event:
        return PairGenerationEvent(self.group_id, self.serial, self._pair, self._files_a, self._files_b)
    
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
