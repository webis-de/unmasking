from event.events import UnmaskingTrainCurveEvent, ConfigurationFinishedEvent, JobFinishedEvent
from event.interfaces import EventHandler, Event
from input.interfaces import SamplePairClass
from output.formats import UnmaskingCurvePlotter
from output.interfaces import Aggregator

import json
import os
from typing import Dict, Any, List, Tuple, Optional


class CurveAverageAggregator(EventHandler, Aggregator):
    """
    Average unmasking curves from multiple runs.

    Handles events: onUnmaskingFinished
    """

    def __init__(self, meta_data: Dict[str, Any] = None):
        """
        :param meta_data: dict with experiment meta data
        """
        self._curves = {}
        self._meta_data = meta_data if meta_data is not None else {}
        self._aggregate_by_class = False

    def handle(self, name: str, event: Event, sender: type):
        if not isinstance(event, UnmaskingTrainCurveEvent):
            raise TypeError("event must be of type UnmaskingTrainCurveEvent")

        self.add_curve(event.pair.pair_id, event.pair.cls, event.values)

    def add_curve(self, identifier: str, cls: SamplePairClass, values: List[float]):
        agg = str(identifier)
        if self._aggregate_by_class:
            agg = str(cls)
            identifier = None

        if agg not in self._curves:
            self._curves[agg] = []

        self._curves[agg].append((str(identifier), str(cls), values))

    def get_aggregated_curves(self) -> Dict[str, Any]:
        avg_curves = {}
        for c in self._curves:
            avg_curves[c] = {}
            if self._aggregate_by_class:
                avg_curves[c]["curve_id"] = self._curves[c][-1][0]
            else:
                avg_curves[c]["cls"] = self._curves[c][-1][1]

            curves = [x for x in zip(*self._curves[c])][2]
            avg_curves[c]["curve"] = [sum(x) / len(x) for x in zip(*curves)]

        return avg_curves

    def save(self, output_dir: str):
        """
        Save accumulated stats to file in JSON format.
        If the file exists, it will be truncated.
        """

        file_name = os.path.join(output_dir, self._get_output_filename_base() + ".json")
        with open(file_name, "w") as f:
            self._meta_data["aggregate_key"] = "class" if self._aggregate_by_class else "curve_id"
            stats = {
                "meta": self._meta_data,
                "curves": self.get_aggregated_curves()
            }
            json.dump(stats, f, indent=2)

    def reset(self):
        self.__init__(self._meta_data)

    @property
    def aggregate_by_class(self):
        """ Whether to aggregate by class (default: False, i.e. aggregate by identifier) """
        return self._aggregate_by_class

    @aggregate_by_class.setter
    def aggregate_by_class(self, agg_by_class: bool):
        self._aggregate_by_class = agg_by_class

    @property
    def meta_data(self) -> Dict[str, Any]:
        """Get experiment meta data"""
        return self._meta_data

    @meta_data.setter
    def meta_data(self, meta_data: Dict[str, Any]):
        """Set experiment meta data"""
        self._meta_data = meta_data


class AggregatedCurvePlotter(UnmaskingCurvePlotter, Aggregator):
    """
    Plot aggregated curves.
    """

    def __init__(self, markers: Dict[SamplePairClass, Tuple[str, str, Optional[str]]] = None,
                 ylim: Tuple[float, float] = (0, 1.0), display: bool = False):
        super().__init__(markers, ylim, display)
        self._aggregators = []

    def handle(self, name: str, event: Event, sender: type):
        if not isinstance(event, ConfigurationFinishedEvent) and not isinstance(event, JobFinishedEvent):
            raise TypeError("event must be of type ConfigurationFinishedEvent or JobFinishedEvent")

        # process all but ourselves, infinite recursion is only moderately cool
        aggregators = [agg for agg in event.aggregators if agg is not self]
        self._aggregators.extend(aggregators)

        for agg in aggregators:
            curve_dict = agg.get_aggregated_curves()
            for curve in curve_dict:
                self.plot_curve(curve_dict[curve]["curve"], curve_dict[curve]["cls"], self.start_new_curve())

    def reset(self):
        super().reset()
        self._aggregators = []

    def get_aggregated_curves(self) -> Dict[str, Any]:
        curves = {}
        for agg in self._aggregators:
            curves.update(agg.get_aggregated_curves())
        return curves

    def add_curve(self, identifier: str, cls: SamplePairClass, values: List[float]):
        raise NotImplementedError()
