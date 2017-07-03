from conf.interfaces import ConfigLoader
from conf.loader import JobConfigLoader
from event.dispatch import EventBroadcaster, MultiProcessEventContext
from event.events import ConfigurationFinishedEvent, JobFinishedEvent
from input.interfaces import SamplePair
from job.interfaces import JobExecutor, ConfigurationExpander
from unmasking.interfaces import UnmaskingStrategy

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import current_process
from time import time
from typing import Any, Dict, Tuple


class ExpandingExecutor(JobExecutor):
    """
    Expanding job executor.

    Expands its job configuration to multiple configurations with various parameter settings
    based on a set of expansion variables. Expansion is performed based on the
    job.experiment.configurations and job.experiment.configuration_expander settings.
    job.experiment.repetitions controls how often each individual configuration is run.

    Multiple runs are aggregated based on the job.experiment.aggregators setting.Events published by this class:

    Events published by this class:

    * `onConfigurationFinished`: [type: ConfigurationFinishedEvent]
                                 fired after an individual configuration has finished execution.
    * `onJobFinished`:           [type JobFinishedEvent]
                                 fired when the job has finished, but before aggregators are asked
                                 to save their outputs
    """
    
    def __init__(self):
        super().__init__()
        self._config = None

    async def run(self, conf: ConfigLoader, output_dir: str = None):
        self._config = conf
        
        self._load_outputs(self._config)
        self._load_aggregators(self._config)
        
        job_id = "job_" + str(int(time()))
        if output_dir is None:
            output_dir = self._config.get("job.output_dir")
        else:
            output_dir = output_dir
        output_dir = os.path.join(output_dir, job_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if not os.path.isdir(output_dir):
            raise IOError("Failed to create output directory '{}', maybe it exists already?".format(output_dir))

        conf.save(os.path.join(output_dir, "job"))

        config_vectors = self._config.get("job.experiment.configurations")
        config_variables = [tuple()]
        expanded_vectors = [tuple()]
        if config_vectors:
            config_expander = self._configure_instance(self._config.get("job.experiment.configuration_expander"))
            if not isinstance(config_expander, ConfigurationExpander):
                raise ValueError("'{}' is not a ConfigurationExpander".format(config_expander.__class__.__name__))

            config_variables = config_vectors.keys()
            expanded_vectors = config_expander.expand(config_vectors.values())

        start_time = time()
        try:
            for config_index, vector in enumerate(expanded_vectors):
                if vector:
                    config_output_dir = os.path.join(output_dir, "config_{:05d}".format(config_index))
                    cfg = JobConfigLoader(self._expand_dict(self._config.get(), config_variables, vector))
                    os.makedirs(config_output_dir)
                    cfg.save(os.path.join(config_output_dir, "job_expanded"))
                else:
                    config_output_dir = output_dir
                    cfg = JobConfigLoader(self._config.get())

                chunk_tokenizer = self._configure_instance(cfg.get("job.input.tokenizer"))
                parser = self._configure_instance(cfg.get("job.input.parser"), chunk_tokenizer)
                iterations = cfg.get("job.experiment.repetitions")

                strat = self._configure_instance(cfg.get("job.unmasking.strategy"))
                loop = asyncio.get_event_loop()
                executor = ProcessPoolExecutor()
                for rep in range(0, iterations):
                    with MultiProcessEventContext:
                        # allow multiprocessing event context to initialize
                        await asyncio.sleep(0)

                        futures = []
                        async for pair in parser:
                            futures.append(loop.run_in_executor(executor, self._exec, strat, pair, cfg))
                        await asyncio.wait(futures)

                    for output in self.outputs:
                        output.save(config_output_dir)
                        output.reset()

                event = ConfigurationFinishedEvent(job_id + "_cfg", config_index, self.aggregators)
                await EventBroadcaster.publish("onConfigurationFinished", event, self.__class__)

            event = JobFinishedEvent(job_id, 0, self.aggregators)
            await EventBroadcaster.publish("onJobFinished", event, self.__class__)

            for aggregator in self.aggregators:
                aggregator.save(output_dir)
                aggregator.reset()
        finally:
            print("Time taken: {:.03f} seconds.".format(time() - start_time))

    def _exec(self, strat: UnmaskingStrategy, pair: SamplePair, cfg: JobConfigLoader):
        """
        Execute actual unmasking strategy.
        This method should be run in a separate process.

        :param strat: unmasking strategy to run
        :param pair: sample pair to run on
        :param cfg: job configuration
        """
        current_process().daemon = True

        sampler = self._configure_instance(cfg.get("job.classifier.sampler"))
        feature_set = self._configure_instance(cfg.get("job.classifier.feature_set"), pair, sampler)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(strat.run(
            pair,
            cfg.get("job.unmasking.iterations"),
            cfg.get("job.unmasking.vector_size"),
            feature_set,
            cfg.get("job.unmasking.relative"),
            cfg.get("job.unmasking.folds"),
            cfg.get("job.unmasking.monotonize")))

    def _expand_dict(self, d: Dict[str, Any], keys: Tuple[str], values: Tuple) -> Dict[str, Any]:
        """
        Expand variables in configuration dictionary.

        :param d: dict to expand
        :param keys: replacement keys
        :param values: expansion values (in the same order as keys)
        :return: expanded dict
        """
        expanded = {}
        for k in d:
            expanded[k] = d[k]

            if type(d[k]) is dict:
                expanded[k] = self._expand_dict(d[k], keys, values)
            elif type(d[k]) is str:
                for repl, val in zip(keys, values):
                    if "$" + repl in d[k]:
                        new_value = d[k].replace("$" + repl, str(val))
                        try:
                            expanded[k] = type(val)(new_value)
                        except (TypeError, ValueError):
                            expanded[k] = new_value
        return expanded
