from event.interfaces import Event


class ProgressEvent(Event):
    """
    Event for indicating progress of an operation with a fixed number of steps to be performed.
    """
    
    def __init__(self, name: str, ops_total: int = 1):
        """
        :param ops_total: total number of operations / steps that will be performed
        """
        super().__init__(name)
        
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