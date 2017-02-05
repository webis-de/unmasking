from abc import ABC, abstractmethod


class FileOutput(ABC):
    """
    Base class for objects which can save their state to a file
    """
    
    @abstractmethod
    def save(self, file_name: str):
        """
        Save object state to given file

        :param file_name: output file name
        """
        pass
