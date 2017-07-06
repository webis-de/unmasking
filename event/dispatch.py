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

from event.interfaces import Event, EventHandler

import asyncio
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Event, JoinableQueue, current_process
from queue import Empty
from sys import stderr
from threading import current_thread
from typing import Set


class EventBroadcaster:
    __subscribers = {}
    
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
        if current_process().name != "MainProcess" or current_thread().name != "MainThread":
            if MultiProcessEventContext.terminate_event.is_set():
                # application is about to terminate, don't accept any new events from workers
                return

            # We are in a worker process, delegate events to main process
            MultiProcessEventContext.queue.put((event_name, event, sender))
            return

        if event_name not in cls.__subscribers:
            return

        for h in cls.__subscribers[event_name]:
            if h[0] is None or sender in h[0]:
                await h[1].handle(event_name, event, sender)


class _MultiProcessEventContextType(type):
    """
    Internal type for providing a multiprocess event manager context.
    """

    queue = JoinableQueue()
    terminate_event = Event()

    _initialized = False

    async def __aenter__(self):
        """
        Initialize event queue consumer thread for multiprocess event handling.
        """

        if self._initialized:
            return

        if current_process().name != "MainProcess" and current_thread().name != "MainThread":
            raise RuntimeError("MultiProcessEventContext must only be opened from main process / thread")

        self._initialized = True
        self.terminate_event.clear()

        asyncio.ensure_future(self.__await_queue())

        # allow multiprocessing event context to initialize
        await asyncio.sleep(0)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Process any remaining events and signal queue consumer thread to exit.
        """

        # wait for remaining events to be processes before exiting the context
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        await loop.run_in_executor(executor, self.queue.join)

        # don't accept new events
        self.terminate_event.set()

        # terminate queue consumer loop (unfortunately, multiprocessing queues don't
        # guarantee proper ordering, so this extra step is necessary)
        self.queue.put(None)
        await asyncio.sleep(0)

        self._initialized = False

    async def __await_queue(self):
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        while loop.is_running():
            f = loop.run_in_executor(executor, self.__wait_queue, self.queue)

            await f
            self.queue.task_done()

            if f.result() is None:
                return

            await EventBroadcaster.publish(*f.result())

    @staticmethod
    def __wait_queue(q):
        """
        Get value of the queue. This method should be called in a separate thread,
        since it blocks until there is an item in the queue.

        :param q: multiprocessing queue
        :return: queue entry
        """
        return q.get(block=True)


class MultiProcessEventContext(metaclass=_MultiProcessEventContextType):
    """
    Context manager for multiprocess event communication.
    Use this in a with statement around any multiprocessing or multithreading
    code to ensure events are properly delegated to the main process / thread.

    You should make sure to "await" as soon as possible after creating the
    context to allow the initializing code to be executed.
    """

    @staticmethod
    def cleanup():
        """
        Cleanup method to ensure all event queues are cleared and worker processes
        are signaled to shut down.

        This method should be called once when shutting down the application.
        """
        # noinspection PyProtectedMember
        if MultiProcessEventContext._initialized:
            print("Shutting down worker processes...", file=stderr)

        MultiProcessEventContext.terminate_event.set()

        try:
            while not MultiProcessEventContext.queue.empty():
                MultiProcessEventContext.queue.get(False)
        except Empty:
            pass

        MultiProcessEventContext.queue.put(None)
