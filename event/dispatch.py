from event.interfaces import Event, EventHandler

from typing import Set


class EventBroadcaster:
    _subscribers = {}
    
    @classmethod
    def subscribe(cls, event_name: str, handler: EventHandler, senders: Set[type] = None):
        """
        Subscribe to events with the name `event_name`.
        When the event is fired, all subscribed :class:`EventHandler`s will be notified
        by calling their :method:`EventHandler.handle()` method.
        
        :param event_name: string identifier of the event to subscribe to
        :param handler: event handler
        :param senders: senders to listen to (None to subscribe to events from all senders)
        """
        if event_name not in cls._subscribers:
            cls._subscribers[event_name] = []
        
        cls._subscribers[event_name].append((senders, handler))
    
    @classmethod
    def unsubscribe(cls, event_name: str, handler, senders: Set[type] = None):
        """
        Unsubscribe `handler` from the given event.
        
        :param event_name: string identifier of the event to unsubscribe from
        :param handler: event handler to unsubscribe
        :param senders: set of senders (must be the same set that was used to subscribe to the event)
        """
        if event_name not in cls._subscribers:
            return
        
        for e in cls._subscribers:
            cls._subscribers[e] = [i for i in cls._subscribers[e] if i != (senders, handler)]
    
    @classmethod
    def publish(cls, event: Event, sender: type):
        """
        Publish the given event and notify all subscribed :class:`EventHandler`s.
        
        :param event: event to publish, which must have its :attr:`Event.name` property set
        :param sender: ``__class__`` type object of the sending class or object
        """
        if event.name not in cls._subscribers:
            return
        
        for h in cls._subscribers[event.name]:
            if h[0] is None or sender in h[0]:
                h[1].handle(event, sender)
