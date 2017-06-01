from conf.interfaces import ConfigLoader
from job.interfaces import JobExecutor

import os
from time import time


class DefaultExecutor(JobExecutor):
    """
    Default job executor.
    """
    
    def __init__(self):
        super().__init__()
        self._config = None

    def run(self, conf: ConfigLoader):
        self._config = conf
        
        chunk_tokenizer = self._configure_instance(self._config.get("job.input.tokenizer"))
        parser = self._configure_instance(self._config.get("job.input.parser"), chunk_tokenizer)
        
        self._load_outputs(self._config)
        self._load_aggregators(self._config)
        
        job_id = "job_" + str(int(time()))
        output_dir = os.path.join(self._config.get("job.output_dir"), job_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if not os.path.isdir(output_dir):
            raise IOError("Failed to create output directory '{}', maybe it exists already?".format(output_dir))
        
        conf.save(os.path.join(output_dir, "job"))
        
        start_time = time()
        try:
            iterations = self._config.get("job.experiment.repetitions")
            
            for rep in range(0, iterations):
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
                    
                for output in self.outputs:
                    output.save(output_dir)
                    output.reset()
                
            for aggregator in self.aggregators:
                aggregator.save(output_dir)
                aggregator.reset()
        finally:
            print("Time taken: {:.03f} seconds.".format(time() - start_time))
