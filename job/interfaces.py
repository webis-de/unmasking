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

from conf.interfaces import ConfigLoader
from event.dispatch import EventBroadcaster
from event.interfaces import EventHandler
from output.interfaces import Output, Aggregator
from util.util import get_base_path

from abc import abstractmethod, ABC
from importlib import import_module
from typing import Any, Dict, Iterable, List, Tuple

import os
import yaml


class JobExecutor(ABC):
    """
    Generic job executor.
    """
    
    def __init__(self):
        self._outputs = []
        self._aggregators = []
    
    @property
    def outputs(self) -> List[Output]:
        """Get configured outputs"""
        return self._outputs
    
    @property
    def aggregators(self) -> List[Aggregator]:
        """Get configured aggregators"""
        return self._aggregators
    
    def _load_class(self, name: str):
        """
        Dynamically load a class based on its fully-qualified module path
        
        :param name: class name
        :return: class
        """
        modules = name.split(".")
        return getattr(import_module(".".join(modules[0:-1])), modules[-1])

    def _configure_instance(self, cfg: Dict[str, Any], *ctr_args):
        """
        Dynamically configure an instance of a class based on the parameters
        defined in the job configuration.
        
        :param cfg: object configuration parameters
        :param ctr_args: constructor arguments
        :return: configured instance
        """
        cls = self._load_class(cfg["name"])
        obj = cls(*ctr_args)

        if "rc_file" in cfg and cfg["rc_file"] is not None:
            rc_file = os.path.join(get_base_path(), cfg["rc_file"])
            with open(rc_file, "r") as f:
                rc_contents = yaml.safe_load(f)

            for rc in rc_contents:
                obj.set_property(rc, rc_contents[rc])

        if "parameters" in cfg and cfg["parameters"] is not None:
            for p in cfg["parameters"]:
                obj.set_property(p, cfg["parameters"][p])
        return obj

    def _subscribe_to_events(self, obj: EventHandler, events: List[Dict[str, Any]]):
        """
        Subscribe an object to events for the given job.

        :param obj: EventHandler object to subscribe
        :param events: list of dicts containing a name key and an optional
                       senders key with a list of allowed senders
        """

        if not isinstance(obj, EventHandler):
            raise ValueError("'{}' is not an EventHandler".format(obj.__class__.__name__))
        
        for event in events:
            senders = None
            if "senders" in event and type(event["senders"]) is list:
                senders = {self._load_class(s) if type(s) is str else s for s in event["senders"]}
            EventBroadcaster.subscribe(event["name"], obj, senders)

    def _load_outputs(self, outputs: List[Dict[str, Any]]):
        """
        Load job output modules.
        
        :param outputs: output settings list
        """
        for output in outputs:
            output_obj = self._configure_instance(output)
            if not isinstance(output_obj, Output):
                raise ValueError("'{}' is not an Output".format(output["name"]))
            
            if "events" in output:
                # noinspection PyTypeChecker
                self._subscribe_to_events(output_obj, output["events"])
            
            self._outputs.append(output_obj)
            
    def _load_aggregators(self, aggs: List[Dict[str, Any]]):
        """
        Load job aggregator modules.
        
        :param aggs: aggregator settings list
        """
        for agg in aggs:
            agg_obj = self._configure_instance(agg)
            if not isinstance(agg_obj, Aggregator):
                raise ValueError("'{}' is not an Aggregator".format(agg["name"]))

            if "events" in agg:
                if not isinstance(agg_obj, EventHandler):
                    raise ValueError("Aggregator '{}' is not an EventHandler".format(agg["name]"]))

                self._subscribe_to_events(agg_obj, agg["events"])

            self._aggregators.append(agg_obj)

    @abstractmethod
    def run(self, conf: ConfigLoader, output_dir: str = None):
        """
        Execute job with given job configuration.

        :param conf: job configuration loader
        :param output_dir: output directory
        """
        pass


class MetaClassificationExecutor(JobExecutor, ABC):
    """
    Base class for meta classification executors.
    """
    pass


class ConfigurationExpander(ABC):
    """
    Base class for configuration expanders.
    """

    @abstractmethod
    def expand(self, configuration_vectors: Iterable[Tuple]) -> Iterable[Tuple]:
        """
        Expand the given configuration vectors based on a certain expansion strategy.

        Generates an iterable sequence of n-dimensional vectors from an input
        Iterable of n configuration vectors where each vector represents a
        single configuration.

        :param configuration_vectors: input vectors with configuration values
        :return: generator of expanded configuration vectors
        """
        pass
