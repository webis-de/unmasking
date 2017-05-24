from conf.interfaces import ConfigLoader
from job.interfaces import JobExecutor

from time import time


class DefaultExecutor(JobExecutor):
    """
    Default job executor.
    """

    def run(self, conf: ConfigLoader):
        start_time = time()
        try:
            pass
        finally:
            print("Time taken: {:.03f} seconds.".format(time() - start_time))
