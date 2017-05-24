from abc import abstractmethod, ABC
from typing import Any


class ConfigLoader(ABC):
    @abstractmethod
    def load(self, filename: str):
        """
        Load configuration from given file.

        :param filename: configuration file name
        """
        pass

    @abstractmethod
    def get(self, name: str) -> Any:
        """
        Get configuration option.

        :param name: name of the option
        :return: option value
        :raise: KeyError if option not found
        """
        pass
