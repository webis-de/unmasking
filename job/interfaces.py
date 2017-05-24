from conf.interfaces import ConfigLoader

from abc import abstractmethod, ABC


class JobExecutor(ABC):
    """
    Generic job executor.
    """

    @abstractmethod
    def run(self, conf: ConfigLoader):
        """
        Execute job with given job configuration.

        :param conf: job configuration loader
        """
        pass
