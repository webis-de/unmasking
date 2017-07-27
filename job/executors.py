# General-purpose unmasking framework
# Copyright (C) 2017 Janek Bevendorff, Webis Group
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

from conf.interfaces import ConfigLoader
from conf.loader import JobConfigLoader
from event.dispatch import EventBroadcaster, MultiProcessEventContext
from event.events import ConfigurationFinishedEvent, JobFinishedEvent
from features.interfaces import FeatureSet
from job.interfaces import JobExecutor, ConfigurationExpander
from meta.interfaces import MetaClassificationModel
from meta.util import unmasking_result_to_numpy
from output.formats import UnmaskingResult
from unmasking.interfaces import UnmaskingStrategy
from util.util import clear_lru_caches

from abc import ABC, abstractmethod
from concurrent.futures import Executor, ProcessPoolExecutor
from time import time
from typing import Any, Dict, Tuple

import asyncio
import os


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

        self._load_outputs(self._config.get("job.outputs"))
        self._load_aggregators(self._config.get("job.experiment.aggregators"))

        job_id, output_dir = self._init_job_output(conf, output_dir)

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
        executor = ProcessPoolExecutor()
        try:
            for config_index, vector in enumerate(expanded_vectors):
                await self._run_configuration(executor, config_index, vector, config_variables, job_id, output_dir)

            event = JobFinishedEvent(job_id, 0, self.aggregators)
            await EventBroadcaster.publish("onJobFinished", event, self.__class__)

            for aggregator in self.aggregators:
                aggregator.save(output_dir)
                aggregator.reset()
        finally:
            executor.shutdown()
            print("Time taken: {:.03f} seconds.".format(time() - start_time))

    async def _run_configuration(self, executor: Executor, config_index: int, vector: Tuple,
                                 config_variables: Tuple[str], job_id: str, output_dir: str):
        """
        Run a single configuration in multiple parallel processes.

        :param executor: ProcessPoolExecutor (or ThreadPoolExecutor) to run the configurations
        :param config_index: index number of the current configuration
        :param vector: vector of expansion values (may be empty)
        :param config_variables: variables to expand with the values from vector
        :param job_id: string id of the running job
        :param output_dir: output directory
        """
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
        repetitions = cfg.get("job.experiment.repetitions")

        strat = self._configure_instance(cfg.get("job.unmasking.strategy"))
        sampler = self._configure_instance(cfg.get("job.classifier.sampler"))
        loop = asyncio.get_event_loop()
        for _ in range(repetitions):
            async with MultiProcessEventContext:
                futures = []

                async for pair in parser:
                    feature_set = self._configure_instance(cfg.get("job.classifier.feature_set"), pair, sampler)
                    futures.append(loop.run_in_executor(executor, self._exec, strat, feature_set, cfg))
                    await asyncio.sleep(0)

                await asyncio.wait(futures)

            for output in self.outputs:
                output.save(config_output_dir)
                output.reset()

        clear_lru_caches()

        event = ConfigurationFinishedEvent(job_id + "_cfg", config_index, self.aggregators)
        await EventBroadcaster.publish("onConfigurationFinished", event, self.__class__)

    @staticmethod
    def _exec(strat: UnmaskingStrategy, feature_set: FeatureSet, cfg: JobConfigLoader):
        """
        Execute actual unmasking strategy.
        This method should be run in a separate process.

        :param strat: unmasking strategy to run
        :param feature_set: feature set for pair
        :param cfg: job configuration
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(strat.run(
                feature_set,
                cfg.get("job.unmasking.iterations"),
                cfg.get("job.unmasking.vector_size"),
                cfg.get("job.unmasking.relative"),
                cfg.get("job.unmasking.folds"),
                cfg.get("job.unmasking.monotonize")))
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.stop()

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


class MetaClassificationExecutor(JobExecutor, ABC):
    """
    Base class for meta classification executors.
    Runs a meta training or classification job on a set of pre-generated unmasking curves.

    Events published by this class:

    * `onJobFinished`: [type JobFinishedEvent]
                       fired when the job has finished
    """

    def __init__(self):
        super().__init__()
        self._config = None

    async def run(self, conf: ConfigLoader, output_dir: str = None):
        job_id, output_dir = self._init_job_output(conf, output_dir)
        model = self._configure_instance(conf.get("job.model"))
        self._load_outputs(conf.get("job.outputs"))

        start_time = time()
        try:
            await self._exec(job_id, model, conf, output_dir)
            event = JobFinishedEvent(job_id, 0, [])
            await EventBroadcaster.publish("onJobFinished", event, self.__class__)

            for output in self.outputs:
                output.save(output_dir)
                output.reset()

        finally:
            print("Time taken: {:.03f} seconds.".format(time() - start_time))

    @abstractmethod
    async def _exec(self, job_id: str, model: MetaClassificationModel, output_dir):
        """
        Execute meta classification task.

        :param job_id: job ID
        :param output_dir: full output directory path
        """
        pass


class MetaTrainExecutor(MetaClassificationExecutor):
    """
    Train and save a meta classification model from given input raw data.
    """

    def __init__(self, input_path: str):
        """
        :param input_path: JSON input file
        """
        super().__init__()
        self._input_path = input_path

    # noinspection PyPep8Naming
    async def _exec(self, job_id, model: MetaClassificationModel, output_dir):
        unmasking_input = UnmaskingResult()
        unmasking_input.load(self._input_path)
        X, y = unmasking_result_to_numpy(unmasking_input)

        if not y:
            raise RuntimeError("Training input must have labels")

        await model.fit(X, y)
        model.save(output_dir)
