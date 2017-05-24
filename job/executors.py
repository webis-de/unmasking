from conf.interfaces import ConfigLoader
from job.interfaces import JobExecutor

from time import time


class DefaultExecutor(JobExecutor):
    """
    Default job executor.
    """
    
    def __init__(self):
        self._config = None

    def run(self, conf: ConfigLoader):
        self._config = conf
        
        chunk_tokenizer = self._configure_instance(self._config.get("job.input.tokenizer"))
        parser = self._configure_instance(self._config.get("job.input.parser"), chunk_tokenizer)
        
        self._subscribe_to_output_events(self._config)
        
        start_time = time()
        try:
            for i, pair in enumerate(parser):
                sampler = self._configure_instance(self._config.get("job.classifier.sampler"))
                feature_set = self._configure_instance(self._config.get("job.classifier.featureSet"), pair, sampler)
                
                unmasking_cfg = self._config.get("job.unmasking")
                strat = self._configure_instance(unmasking_cfg["strategy"])
                strat.run(
                    unmasking_cfg["iterations"],
                    unmasking_cfg["vectorSize"],
                    feature_set,
                    unmasking_cfg["relative"],
                    unmasking_cfg["folds"],
                    unmasking_cfg["monotonize"])
        finally:
            print("Time taken: {:.03f} seconds.".format(time() - start_time))

