from abc import ABC, abstractmethod


class Event:
    """
    Base class for events.
    :class:`EventHandler`s can subscribe to individual events.
    """
    
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self):
        """Get event identifier string."""
        return self._name
    
    @name.setter
    def name(self, value):
        """Set event identifier string."""
        self._name = value


class EventHandler(ABC):
    """
    Base class for :class:`Event` subscribers.
    """
    
    @abstractmethod
    def handle(self, event: Event, sender: str):
        """
        Handle the given event.
        
        :param event: fired event
        :param sender: name of the sender
        """
        pass
