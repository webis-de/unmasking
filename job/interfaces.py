from abc import abstractmethod, ABC
from importlib import import_module
from typing import Any, Dict

from conf.interfaces import ConfigLoader
from event.dispatch import EventBroadcaster


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


