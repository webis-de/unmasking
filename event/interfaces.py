from abc import ABC, abstractmethod


class Event:
    """
    Base class for events.
    :class:`EventHandler`s can subscribe to individual events.
    """
    pass


class EventHandler(ABC):
    """
    Base class for :class:`Event` subscribers.
    """
    
    @abstractmethod
    def handle(self, name: str, event: Event, sender: type):
        """
        Handle the given event.
        
        :param name: name of the fired event
        :param event: the fired event itself
        :param sender: class of the sender
        """
        pass
