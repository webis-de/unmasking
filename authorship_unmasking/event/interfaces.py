# Copyright (C) 2017-2019 Janek Bevendorff, Webis Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta, abstractmethod
from uuid import UUID, uuid5
from typing import Iterable


class Event:
    """
    Base class for events.
    Events have a group ID for identifying an event as part of a single process / group of events
    and a serial number for identifying and ordering individual events within a group.

    :class:: `EventHandler`s can subscribe to individual events.
    """

    EVENT_NS = UUID("fec851d1-2876-5072-b07b-cb289dd5dbef")

    def __init__(self, group_id: str, serial: int):
        """
        :param group_id: string token identifying this event as part of a group of events
        :param serial: serial number for identifying and ordering individual events within a group
        """
        self._group_id = group_id
        self._serial = serial

    @classmethod
    def new_event(cls, previous: "Event") -> "Event":
        """
        Instantiate a new event in the same group as the `previous` event, but
        with incremented serial number. Uses :meth:: clone() to create an
        event instance copy.

        `previous` must be an instance of the same `Event` class.

        :param previous: previous event in the same group
        :return: new event with the same group ID and incremented serial number
        """
        if previous is None or not isinstance(previous, cls):
            raise ValueError("Previous event must be an instance of class '{}'".format(cls.__name__))

        clone = previous.clone()
        clone._serial += 1
        return clone

    def clone(self) -> "Event":
        """
        Return a new cloned instance of this event.
        Sub classes should override this method if they change the constructor signature.
        """
        event = self.__class__(self.group_id, self.serial)
        event.__dict__ = self.__dict__.copy()
        return event

    @property
    def group_id(self) -> str:
        """Get group ID token"""
        return self._group_id

    @property
    def serial(self) -> int:
        """Get serial number"""
        return self._serial

    @classmethod
    def generate_group_id(cls, sources: Iterable[str]) -> str:
        """
        Generate a UUID string that can be used as a group ID for events based
        on a given set of event sources. The UUID string will be unique
        based on the event subclass and the set of sources.
        The order of the given source strings doesn't matter.

        :param sources: list of sources (e.g. input file names) which events in
                        this group are generated from
        """
        sources_str = cls.__name__ + ":" + ",".join(sorted(sources))
        return str(uuid5(cls.EVENT_NS, sources_str))


class EventHandler(metaclass=ABCMeta):
    """
    Base class for :class:`Event` subscribers.
    """
    
    @abstractmethod
    async def handle(self, name: str, event: Event, sender: type):
        """
        Handle the given event asynchronously.
        
        :param name: name of the fired event
        :param event: the fired event itself
        :param sender: class of the sender
        """
        pass
