from event.interfaces import Event

from typing import List, Optional, Tuple


class ProgressEvent(Event):
    """
    Event for indicating progress of an operation with a fixed number of steps to be performed.
    """
    
    def __init__(self, ops_total: int = 1):
        """
        :param ops_total: total number of operations / steps that will be performed
        """
        super().__init__()
        
        if ops_total < 1:
            raise AttributeError("ops_total must be greater than 1")
        
        self._ops_done = 0
        self._ops_total = ops_total
    
    @property
    def ops_total(self) -> int:
        """Get total number of operations / steps that will be performed."""
        return self._ops_total
    
    @property
    def ops_done(self) -> int:
        """Get number of operations that have been completed so far."""
        return self._ops_done
    
    @ops_done.setter
    def ops_done(self, new_val: int):
        """Update number of operations that have been completed."""
        if new_val > self._ops_total:
            raise AttributeError("ops_done cannot be larger than total number of operations")
        self._ops_done = new_val
    
    @property
    def percent_done(self) -> float:
        """Total progress in percent."""
        return (float(self._ops_done) / self.ops_total) * 100.0
    
    @property
    def finished(self) -> bool:
        """True if all operations have finished."""
        return self._ops_done >= self._ops_total
    
    def increment(self, steps: int = 1):
        """
        Increment completed operations counter.
        If the counter would exceed the set total number of operations, no further increments will be added.
        
        :param steps: how many steps to increment
        """
        self._ops_done = min(self._ops_done + steps, self._ops_total)


class UnmaskingTrainCurveEvent(Event):
    """
    Event for updating training curves of pairs during unmasking.
    """
    
    def __init__(self, n: int = 0, pair=None, feature_set: type = None):
        """
        :param n: predicted final number of total values (should be set to the total number of unmasking iterations)
        :param pair: pair for which this curve is being calculated
        :param feature_set: feature set class used for generating this curve
        """
        super().__init__()
        self._n = n
        self._values = []
        self._pair = pair
        self._feature_set = feature_set
    
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
    
    def __init__(self, pair=None, files_a: Optional[List[str]] = None, files_b: Optional[List[str]] = None):
        """
        :param pair: pair for which this event is emitted
        :param files_a: participating files for chunk set a
        :param files_b: participating files for chunk set b
        """
        self._pair = pair
        self._files_a = [] if files_a is None else files_a
        self._files_b = [] if files_b is None else files_b
    
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
