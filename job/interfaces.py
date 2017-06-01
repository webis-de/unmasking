from conf.interfaces import ConfigLoader
from event.dispatch import EventBroadcaster
from event.interfaces import EventHandler
from output.interfaces import Output, Aggregator

from abc import abstractmethod, ABC
from importlib import import_module
from typing import Any, Dict, List


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

    def _load_outputs(self, conf: ConfigLoader):
        """
        Load job output modules.
        
        :param conf: job configuration
        """
        outputs = conf.get("job.outputs")
    
        for output in outputs:
            format_obj = self._configure_instance(output)
            if not isinstance(format_obj, Output):
                raise ValueError("'{}' is not an Output".format(output["name"]))
            
            if "events" in output:
                # noinspection PyTypeChecker
                self._subscribe_to_events(format_obj, output["events"])
            
            self._outputs.append(output)
            
    def _load_aggregators(self, conf: ConfigLoader):
        """
        Load job aggregator modules.
        
        :param conf: job configuration
        """
        aggs = conf.get("job.experiment.aggregators")
    
        for output in aggs:
            format_obj = self._configure_instance(output)
            if not isinstance(format_obj, Aggregator):
                raise ValueError("'{}' is not an Aggregator".format(output["name"]))
            
            if "events" in output:
                # noinspection PyTypeChecker
                self._subscribe_to_events(format_obj, output["events"])
            
            self._aggregators.append(output)
    
    @abstractmethod
    def run(self, conf: ConfigLoader):
        """
        Execute job with given job configuration.

        :param conf: job configuration loader
        """
        pass


