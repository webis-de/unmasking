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
from event.events import *
from features.interfaces import ChunkSampler, FeatureSet
from job.interfaces import JobExecutor, ConfigurationExpander, Strategy
from input.interfaces import CorpusParser, Tokenizer
from meta.interfaces import MetaClassificationModel
from output.formats import UnmaskingResult
from util.util import clear_lru_caches

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from concurrent.futures import Executor, ProcessPoolExecutor
from glob import glob
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from time import time
from typing import Any, Dict, Union, Tuple

import asyncio
import numpy as np
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

    async def run(self, conf: ConfigLoader, output_dir: str = None):
        self._config = conf

        self._load_outputs(self._config.get("job.outputs"))
        self._load_aggregators(self._config.get("job.experiment.aggregators"))

        job_id, output_dir = self._init_job_output(conf, output_dir)

        config_vectors = self._config.get("job.experiment.configurations")
        config_variables = [tuple()]
        expanded_vectors = [tuple()]
        if config_vectors:
            config_expander = self._configure_instance(
                self._config.get("job.experiment.configuration_expander"), ConfigurationExpander)

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
                await aggregator.save(output_dir)
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

        chunk_tokenizer = self._configure_instance(cfg.get("job.input.tokenizer"), Tokenizer)
        parser = self._configure_instance(cfg.get("job.input.parser"), CorpusParser, (chunk_tokenizer,))
        repetitions = cfg.get("job.experiment.repetitions")
        strat = self._configure_instance(cfg.get("job.exec.strategy"), Strategy)
        sampler = self._configure_instance(cfg.get("job.classifier.sampler"), ChunkSampler)

        loop = asyncio.get_event_loop()
        for _ in range(repetitions):
            async with MultiProcessEventContext:
                futures = []

                async for pair in parser:
                    feature_set = self._configure_instance(
                        cfg.get("job.classifier.feature_set"), FeatureSet, (pair, sampler))
                    futures.append(loop.run_in_executor(executor, self._exec, strat, feature_set))
                    await asyncio.sleep(0)

                await asyncio.wait(futures)

            for output in self.outputs:
                await output.save(config_output_dir)
                output.reset()

        clear_lru_caches()

        event = ConfigurationFinishedEvent(job_id + "_cfg", config_index, self.aggregators)
        await EventBroadcaster.publish("onConfigurationFinished", event, self.__class__)

    @staticmethod
    def _exec(strat: Strategy, feature_set: FeatureSet):
        """
        Execute actual unmasking strategy on a pair feature set.
        This method should be run in a separate process.

        :param strat: unmasking strategy to run
        :param feature_set: feature set for pair
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(strat.run(feature_set))
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.stop()

    def _expand_dict(self, d: Dict[str, Any], keys: Tuple[str], values: Tuple) -> Dict[str, Any]:
        """
        Expand configuration variables.

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


class MetaClassificationExecutor(JobExecutor, metaclass=ABCMeta):
    """
    Base class for meta classification executors.
    Runs a meta training or classification job on a set of pre-generated unmasking curves.

    Events published by this class:

    * `onJobFinished`: [type JobFinishedEvent]
                       fired when the job has finished
    * `onModelFit`:    [type ModelFitEvent]
                       fired when model has been successfully fit to a dataset
    """

    def __init__(self):
        super().__init__()

    async def run(self, conf: ConfigLoader, output_dir: str = None):
        self._config = conf
        job_id, output_dir = self._init_job_output(conf, output_dir)
        self._load_outputs(conf.get("job.outputs"))

        start_time = time()
        try:
            await self._exec(job_id, output_dir)
            event = JobFinishedEvent(job_id, 0, [])
            await EventBroadcaster.publish("onJobFinished", event, self.__class__)

            for output in self.outputs:
                await output.save(output_dir)
                output.reset()

        finally:
            print("Time taken: {:.03f} seconds.".format(time() - start_time))

    @abstractmethod
    async def _exec(self, job_id: str, output_dir):
        """
        Execute meta classification task.

        :param job_id: job ID
        :param output_dir: full output directory path
        """
        pass

    # noinspection PyPep8Naming
    async def _train_from_json(self, input_path: str, model: MetaClassificationModel):
        """
        Train a new model from given JSON input data.

        :param input_path: path to the input JSON file
        :param model: model to train
        """
        unmasking = UnmaskingResult()
        unmasking.load(input_path)
        X, y = unmasking.to_numpy()
        if y is None:
            raise RuntimeError("Training input must have labels")

        await model.optimize(X, y)
        X, y = await model.fit(X, y)

        y = [unmasking.numpy_label_to_str(l) for l in y]
        event = ModelFitEvent(input_path, 0, X, y)
        await EventBroadcaster.publish("onModelFit", event, self.__class__)

    @staticmethod
    def c_at_1_score(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]):
        """
        Return c@1 score of prediction.
        See Peñas and Rodrigo, 2011: A Simple Measure to Assess Non-response

        :param y_true: true labels
        :param y_pred: predicted labels, -1 for non-decisions
        :return: c@1 score
        """
        n = len(y_true)
        n_ac = 0
        n_u = 0

        for i, pred in enumerate(y_pred):
            if pred <= -1:
                n_u += 1
            elif pred == y_true[i]:
                n_ac += 1

        return 1.0 / n * (n_ac + (n_ac / n) * n_u)

    @staticmethod
    def f_05_u_score(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray], pos_label: int):
        """
        Return F0.5u score of prediction.

        :param y_true: true labels
        :param y_pred: predicted labels, -1 for non-decisions
        :param pos_label: positive class label
        :return: F0.5u score
        """

        n_tp = 0
        n_fn = 0
        n_fp = 0
        n_u = 0

        for i, pred in enumerate(y_pred):
            if pred <= -1:
                n_u += 1
            elif pred == pos_label and pred == y_true[i]:
                n_tp += 1
            elif pred == pos_label and pred != y_true[i]:
                n_fp += 1
            elif y_true[i] == pos_label and pred != y_true[i]:
                n_fn += 1

        return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)


class MetaTrainExecutor(MetaClassificationExecutor):
    """
    Train and save a meta classification model from given input raw data.

    Events published by this class:

    * `onJobFinished`: [type JobFinishedEvent]
                       fired when the job has finished
    * `onModelFit`:    [type ModelFitEvent]
                       fired when model has been successfully fit to a dataset
    """

    def __init__(self, input_path: str):
        """
        :param input_path: JSON input file
        """
        super().__init__()
        self._input_path = input_path

    async def _exec(self, job_id, output_dir):
        if not self._input_path.endswith(".json"):
            raise ValueError("Input file must be JSON")

        model = self._configure_instance(self._config.get("job.model"), MetaClassificationModel)
        await self._train_from_json(self._input_path, model)
        await model.save(output_dir)


class MetaApplyExecutor(MetaClassificationExecutor):
    """
    Apply a pre-trained model to a test data set.

    Events published by this class:

    * `onJobFinished`:    [type JobFinishedEvent]
                          fired when the job has finished
    * `onModelFit`:       [type ModelFitEvent]
                          fired when model has been successfully fit to a dataset
    * `onDataPredicted`:  [type ModelPredictEvent]
                          fired when model has been applied to dataset to predict samples
    """

    def __init__(self, model: Union[str, MetaClassificationModel], test_data: Union[str, UnmaskingResult]):
        """
        :param model: pre-trained model or path to model file
        :param test_data: test data set or path to raw JSON file
        """
        super().__init__()
        if type(model) == str:
            self._model = None
            self._model_path = model
        else:
            self._model = model
            self._model_path = None

        if type(model) == str:
            self._test_data = None
            self._test_data_path = test_data
        else:
            self._test_data = test_data
            self._test_data_path = None

    async def _load_data(self):
        """
        Load training and test data from files if needed.
        """
        # load model from model or JSON file if path is given
        if self._model_path:
            self._model = self._configure_instance(self._config.get("job.model"), MetaClassificationModel)
            if self._model_path.endswith(".json"):
                await self._train_from_json(self._model_path, self._model)
            else:
                await self._model.load(self._model_path)
            self._model_path = None

        # load test data from file if path is given
        if self._test_data_path:
            self._test_data = UnmaskingResult()
            self._test_data.load(self._test_data_path)
            self._test_data_path = None

    # noinspection PyPep8Naming
    async def _predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform actual prediction and update stored :class:: UnmaskingResult.

        :param X: data points
        :return: predicted classes as int vector and decision probabilities
        """
        pred = await self._model.predict(X)
        decision_func = await self._model.decision_function(X)
        prob = np.abs(decision_func)
        prob = (prob - np.min(prob)) / (np.max(prob) - np.min(prob))

        for i, curve_id in enumerate(self._test_data.curves):
            if pred[i] > -1:
                pred_cls = self._test_data.numpy_label_to_str(pred[i])
                self._test_data.add_prediction(curve_id, pred_cls, prob[i])
            else:
                self._test_data.add_prediction(curve_id, None, None)

        return pred, prob

    async def _exec(self, job_id, output_dir):
        await self._load_data()

        # noinspection PyPep8Naming
        X, _ = self._test_data.to_numpy()
        y, _ = await self._predict(X)

        # noinspection PyPep8Naming
        X_filtered = [x for i, x in enumerate(X) if y[i] > -1]
        y_filtered = [self._test_data.numpy_label_to_str(l) for l in y if l > -1]
        event = ModelPredictEvent(job_id, 0, X_filtered, y_filtered, False)
        await EventBroadcaster.publish("onDataPredicted", event, self.__class__)

        await self._test_data.save(output_dir)


class MetaEvalExecutor(MetaApplyExecutor):
    """
    Evaluate model quality against a labeled test set.

    Events published by this class:

    * `onJobFinished`:    [type JobFinishedEvent]
                          fired when the job has finished
    * `onDataPredicted`:  [type ModelMetricsEvent]
                          fired when model has been applied to dataset to predict samples
    """

    # noinspection PyPep8Naming
    async def _exec(self, job_id, output_dir):
        await self._load_data()

        X, y = self._test_data.to_numpy()
        if y is None:
            raise ValueError("Test set must have labels")
        pred, _ = await self._predict(X)

        # assume positive class is class with highest int label (usually 1)
        # TODO: let user choose different positive class
        positive_cls = np.max(y)
        negative_cls = np.min(y)

        # eliminate all non-decisions
        y_pred_filtered = pred[pred > -1]
        y_pred_all      = np.copy(pred)
        y_pred_all[y_pred_all == -1] = negative_cls
        y_filtered     = y[pred > -1]
        y_actual_str   = [self._test_data.numpy_label_to_str(y) for y in y_filtered]
        X_filtered     = [x for i, x in enumerate(X) if pred[i] > -1]

        metrics = OrderedDict((
            ("accuracy", accuracy_score(y_filtered, y_pred_filtered)),
            ("c_at_1", self.c_at_1_score(y, pred)),
            ("frac_classified", len(y_pred_filtered) / len(y)),
        ))

        if len(self._test_data.meta["classes"]) == 2:
            # binary classification
            metrics.update(OrderedDict((
                ("f1", f1_score(y_filtered, y_pred_filtered, pos_label=positive_cls, average="binary")),
                ("precision", precision_score(y_filtered, y_pred_filtered, pos_label=positive_cls, average="binary")),
                ("recall", recall_score(y_filtered, y_pred_filtered, pos_label=positive_cls, average="binary")),
                ("recall_total", recall_score(y, y_pred_all, pos_label=positive_cls, average="binary")),
                ("f_05_u", self.f_05_u_score(y, pred, pos_label=positive_cls)),
                ("positive_cls", self._test_data.numpy_label_to_str(positive_cls))
            )))
        else:
            # multi-class classification
            metrics.update(OrderedDict((
                ("f1", f1_score(y_filtered, y_pred_filtered, average="weighted")),
                ("precision", precision_score(y_filtered, y_pred_filtered, average="weighted")),
                ("recall", recall_score(y_filtered, y_pred_filtered, average="weighted")),
                ("recall_total", recall_score(y, y_pred_all, average="weighted")),
            )))

        self._test_data.meta["params"] = self._model.params
        self._test_data.meta["metrics"] = metrics

        event = ModelMetricsEvent(job_id, 0, X_filtered, y_actual_str, True, metrics)
        await EventBroadcaster.publish("onDataPredicted", event, self.__class__)

        await self._test_data.save(output_dir)


class MetaModelSelectionExecutor(MetaClassificationExecutor):
    """
    Evaluate quality of different unmasking models using cross validation.
    Cross-validates all configurations generated by a ::class:ExpandingExecutor and
    selects the best-performing one.

    Events published by this class:

    * `onJobFinished`:    [type JobFinishedEvent]
                          fired when the job has finished
    """

    def __init__(self, configurations_folder: str, folds: int = 10):
        """
        :param configurations_folder: input folder containing run results with different configurations / hyperparameters
        :param metric: metric by which to find the best model
        """
        super().__init__()
        self._folds = folds
        self._input_configs = [f for f in glob(os.path.join(configurations_folder, "config_*")) if os.path.isdir(f)]

    # noinspection PyPep8Naming
    async def _exec(self, job_id, output_dir):
        agg_conf = self._config.get("job.model_selection.aggregator")
        best_model = (-1.0, None, "")
        scorer = make_scorer(self.c_at_1_score)

        if not self._input_configs:
            return

        for conf_folder in self._input_configs:
            agg = self._configure_instance(agg_conf, assert_type=Aggregator)

            for result_path in glob(os.path.join(conf_folder, "*Accumulator.*.json")):
                r = UnmaskingResult()
                r.load(result_path)
                curves = r.curves
                for c in curves:
                    agg.add_curve(c, curves[c]["cls"], curves[c]["values"])

            X, y = agg.get_aggregated_output().to_numpy()
            if y is None:
                raise ValueError("Test set must have labels")

            model = self._configure_instance(self._config.get("job.model"),
                                             MetaClassificationModel)    # type: MetaClassificationModel
            estimator = model.get_configured_estimator()
            cv_scores = cross_val_score(estimator, X, y, scoring=scorer, cv=self._folds, n_jobs=-1)
            cv_score = float(np.mean(cv_scores))

            if cv_score > best_model[0]:
                best_model = (cv_score, agg, conf_folder)

            event = UnmaskingModelEvaluatedEvent(job_id, 0, conf_folder, cv_score)
            await EventBroadcaster.publish("onUnmaskingModelEvaluated", event, self.__class__)

        event = UnmaskingModelSelectedEvent(job_id, 0, best_model[2], best_model[0],
                                            best_model[1].get_aggregated_output())
        await EventBroadcaster.publish("onUnmaskingModelSelected", event, self.__class__)
        await best_model[1].save(output_dir)
