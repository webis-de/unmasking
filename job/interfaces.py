from conf.interfaces import ConfigLoader
from event.dispatch import EventBroadcaster

from abc import abstractmethod, ABC
from importlib import import_module
from typing import Any, Dict


class JobExecutor(ABC):
    """
    Generic job executor.
    """
    
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

    def _subscribe_to_output_events(self, conf: ConfigLoader):
        """
        Subscribe to output events for the given job.
        
        :param conf: job configuration
        """
        outputs = conf.get("job.outputs")
    
        for output in outputs:
            format_obj = self._configure_instance(output)
            senders = None
            if "senders" in output and type(output["senders"]) is list:
                senders = {self._load_class(s) for s in output["senders"]}
            for event in output["events"]:
                EventBroadcaster.subscribe(event, format_obj, senders)
    
    @abstractmethod
    def run(self, conf: ConfigLoader):
        """
        Execute job with given job configuration.

        :param conf: job configuration loader
        """
        pass


class Configurable:
    """
    Base class for classes which are configurable via @properties.
    """
    
    def set_property(self, name: str, value: Any):
        """
        Dynamically set a given configuration property.
        
        :param name: property name
        :param value: property value
        :raise: KeyError if property does not exist
        """
        if not self.has_property(name):
            raise KeyError("{}@{}: No such configuration property".format(self.__class__.__name__, name))

        setattr(self, name, value)
    
    def has_property(self, name: str) -> bool:
        """
        Check whether a class has a given property and if is of type property.
        
        :param name: property name
        :return: Whether object has a given property
        """
        return hasattr(self.__class__, name) and isinstance(getattr(self.__class__, name), property)
