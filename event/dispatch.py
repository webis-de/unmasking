from event.interfaces import Event, EventHandler

import asyncio
from multiprocessing import Lock, Queue, current_process
from threading import current_thread
from typing import Set


class EventBroadcaster:
    __subscribers = {}
    __lock = Lock()
    __initialized = False
    __queue = Queue()

    @classmethod
    def init_multiprocessing_queue(cls):
        """
        Initialize queue for multiprocess communication.
        If you plan to send events from other threads or processes, you need to call
        this method once from the main process.
        """
        if cls.__initialized:
            return

        if current_process().name != "MainProcess" and current_thread().name != "MainThread":
            raise RuntimeError("init_multiprocessing_queue() must only be called from main process")

        asyncio.ensure_future(cls.__wait_for_queue(asyncio.get_event_loop()))

    @classmethod
    async def __wait_for_queue(cls, loop: asyncio.AbstractEventLoop):
        """
        Coroutine to wait for multiprocessing Queue.
        """
        while True:
            params = await loop.run_in_executor(None, cls.__queue.get, True)
            await cls.publish(*params)
    
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
        if event_name not in cls.__subscribers:
            cls.__subscribers[event_name] = []

        cls.__subscribers[event_name].append((senders, handler))
    
    @classmethod
    def unsubscribe(cls, event_name: str, handler, senders: Set[type] = None):
        """
        Unsubscribe `handler` from the given event.
        
        :param event_name: string identifier of the event to unsubscribe from
        :param handler: event handler to unsubscribe
        :param senders: set of senders (must be the same set that was used to subscribe to the event)
        """
        if event_name not in cls.__subscribers:
            return

        for e in cls.__subscribers:
            cls.__subscribers[e] = [i for i in cls.__subscribers[e] if i != (senders, handler)]
    
    @classmethod
    async def publish(cls, event_name: str, event: Event, sender: type):
        """
        Publish the given event and notify all subscribed :class:`EventHandler`s.
        This method is thread-safe. If this method is called from a worker process,
        calls will be delegated to the main process.
        
        :param event_name: name of this event (e.g. 'onProgress')
                           The name can be freely chosen, but should start with 'on' and
                           use camelCasing to separate words
        :param event: event to publish, which must have its :attr:`Event.name` property set
        :param sender: ``__class__`` type object of the sending class or object
        """
        try:
            cls.__lock.acquire()

            if current_process().name != "MainProcess":
                cls.__queue.put((event_name, event, sender))
                return

            if event_name not in cls.__subscribers:
                return

            for h in cls.__subscribers[event_name]:
                if h[0] is None or sender in h[0]:
                    await h[1].handle(event_name, event, sender)
        finally:
            cls.__lock.release()
