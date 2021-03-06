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

from authorship_unmasking.conf.interfaces import ConfigLoader, Configurable
from authorship_unmasking.event.dispatch import EventBroadcaster
from authorship_unmasking.event.interfaces import EventHandler
from authorship_unmasking.features.interfaces import FeatureSet
from authorship_unmasking.output.interfaces import Output, Aggregator

from abc import abstractmethod, ABCMeta
from importlib import import_module
from time import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
import yaml


class JobExecutor(metaclass=ABCMeta):
    """
    Generic job executor.
    """

    def __init__(self):
        self._outputs = []
        self._aggregators = []
        self._config = None     # type: ConfigLoader

    @property
    def outputs(self) -> List[Output]:
        """Get configured outputs"""
        return self._outputs

    @property
    def aggregators(self) -> List[Aggregator]:
        """Get configured aggregators"""
        return self._aggregators

    def _init_job_output(self, conf: ConfigLoader, output_dir: str = None) -> Tuple[str, Optional[str]]:
        """
        Initialize job output directory and return job ID.
        If `output_dir` is not set, the `job.output_dir` directive provided by
        the given :class:: ConfigLoader will be used.

        :param conf: config loader
        :param output_dir: base directory to save job outputs to
        :return: tuple of generated job ID and absolute output directory path
        """
        job_id = "job_" + str(int(time()))

        output_dir = conf.get("job.output_dir") if not output_dir else output_dir
        if output_dir:
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(conf.get_config_path(), output_dir)

            output_dir = os.path.relpath(os.path.join(output_dir, job_id))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if not os.path.isdir(output_dir):
                raise IOError("Failed to create output directory '{}', maybe it exists already?".format(output_dir))

            conf.save(os.path.join(output_dir, "job"))

        return job_id, output_dir

    def _load_class(self, name: str):
        """
        Dynamically load a class based on its fully-qualified module path
        
        :param name: class name
        :return: class
        """
        modules = name.split(".")
        mod_path = ".".join(modules[0:-1])
        mod_name = modules[-1]
        try:
            return getattr(import_module(mod_path), mod_name)
        except ModuleNotFoundError:
            return getattr(import_module('authorship_unmasking.' + mod_path), mod_name)

    def _configure_instance(self, cfg: Dict[str, Any], assert_type: type = None, ctr_args: Iterable[Any] = None):
        """
        Dynamically configure an instance of a class based on the parameters
        defined in the job configuration.
        
        :param cfg: object configuration parameters
        :param assert_type: raise exception if object is not of this type
        :param ctr_args: constructor arguments
        :return: configured instance
        """
        cls = self._load_class(cfg["name"])
        if ctr_args is None:
            obj = cls()
        else:
            obj = cls(*ctr_args)

        if assert_type is not None:
            self._assert_type(obj, assert_type)

        params = {}
        if "rc_file" in cfg and cfg["rc_file"]:
            rc_file = self._config.resolve_relative_path(cfg["rc_file"])
            with open(rc_file, "r") as f:
                params.update(yaml.safe_load(f))

        if "parameters" in cfg and cfg["parameters"]:
            params.update(cfg["parameters"])

        for p in params:
            val = params[p]
            if not obj.has_property(p):
                continue

            if type(val) is str and obj.is_path_property(p):
                val = self._config.resolve_relative_path(os.path.join('..', val))
            elif obj.is_instance_property(p):
                is_list = obj.is_instance_list_property(p)
                if is_list and obj.delegate_args:
                    val = [self._configure_instance(v, assert_type, ctr_args) for v in val]
                elif is_list:
                    val = [self._configure_instance(v) for v in val]
                elif obj.delegate_args:
                    val = self._configure_instance(val, assert_type, ctr_args)
                else:
                    val = self._configure_instance(val)

            obj.set_property(p, val)

        return obj

    def _subscribe_to_events(self, obj: EventHandler, events: List[Dict[str, Any]]):
        """
        Subscribe an object to events for the given job.

        :param obj: EventHandler object to subscribe
        :param events: list of dicts containing a name key and an optional
                       senders key with a list of allowed senders
        """
        self._assert_type(obj, EventHandler)

        for event in events:
            senders = None
            if "senders" in event and type(event["senders"]) is list:
                senders = {self._load_class(s) if type(s) is str else s for s in event["senders"]}
            EventBroadcaster().subscribe(event["name"], obj, senders)

    def _load_outputs(self, outputs: List[Dict[str, Any]]):
        """
        Load job output modules.

        :param outputs: output settings list
        """
        for output in outputs:
            output_obj = self._configure_instance(output, assert_type=Output)

            if "events" in output and output["events"]:
                self._subscribe_to_events(output_obj, output["events"])

            self._outputs.append(output_obj)

    def _load_aggregators(self, aggs: List[Dict[str, Any]]):
        """
        Load job aggregator modules.

        :param aggs: aggregator settings list
        """
        for agg in aggs:
            agg_obj = self._configure_instance(agg, assert_type=Aggregator)

            if "events" in agg and agg["events"]:
                self._subscribe_to_events(agg_obj, agg["events"])

            self._aggregators.append(agg_obj)

    def _assert_type(self, obj: object, t: type):
        """
        Assert an object to be an instance of a certain class, otherwise raise an exception.

        :param obj: object
        :param t: type
        """
        if not isinstance(obj, t):
            raise ValueError("'{}.{}' is not a subclass of {}".format(
                obj.__class__.__module__, obj.__class__.__name__, t.__name__))

    @abstractmethod
    def run(self, conf: ConfigLoader, output_dir: str = None):
        """
        Execute job with given job configuration.

        :param conf: job configuration loader
        :param output_dir: output directory
        """
        pass


class ConfigurationExpander(metaclass=ABCMeta):
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


class Strategy(Configurable, metaclass=ABCMeta):
    """
    Base class for execution strategies.
    """

    @abstractmethod
    async def run(self, fs: FeatureSet):
        """
        Run execution strategy.

        :param fs: parametrized feature set to execute on
        """
