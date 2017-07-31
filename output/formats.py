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

from event.events import *
from event.interfaces import EventHandler
from input.interfaces import SamplePairClass
from output.interfaces import Output

from collections import OrderedDict
from random import randint
from typing import Any, Dict, Union

import asyncio
import gc
import json
import matplotlib
import matplotlib.ticker
import numpy as np
import os
import sys

# don't use default Qt backend if we are operating without a display server
if sys.platform == "linux" and os.environ.get("DISPLAY") is None:
    matplotlib.use("Agg")

try:
    import matplotlib.pyplot as pyplot
except (ModuleNotFoundError, ImportError):
    matplotlib.use("Agg", warn=False, force=True)
    import matplotlib.pyplot as pyplot


class UnmaskingResult(Output):
    """
    Unmasking result DTO.
    """

    def __init__(self):
        self._meta = OrderedDict()
        self._curves = OrderedDict()
        self._classes = set()
        self._classes_mapping = None
        self._inv_classes_mapping = None

    @property
    def curves(self) -> Dict[str, Any]:
        """Get curves as ordered dict"""
        return self._curves

    @property
    def meta(self) -> Dict[str, Any]:
        """Get meta data as ordered dict"""
        self._meta["classes"] = sorted(self._classes)
        return self._meta

    def add_curve(self, curve_id: str, cls: Optional[str], values: List[float],
                  files: List[Union[List[str], str]], **kwargs):
        """
        Add curve to result.

        :param curve_id: string curve ID
        :param cls: pair class (None if unknown)
        :param values: curve data points
        :param files: participating files (can be a list of lists to separate files of a pair into buckets)
        :param kwargs: further properties to add to the curve
        """
        self._curves[curve_id] = OrderedDict([
            ("cls", cls),
            ("values", values),
            ("files", files)
        ])
        if kwargs:
            self._curves[curve_id].update(kwargs)

        self._classes.add(cls)

    def add_prediction(self, curve_id: str, cls: Optional[str], prob: Optional[float]):
        """
        Add a prediction to a curve.

        :param curve_id: curve to add the prediction to
        :param cls: predicted class (None if no decision has been made)
        :param prob: prediction certainty / probability (None if no decision has been made)
        """
        if cls is None:
            pred = None
        else:
            pred = OrderedDict([
                ("cls", cls),
                ("prob", prob)
            ])

        if "pred" not in self._curves[curve_id]:
            self._curves[curve_id]["pred"] = []
        self._curves[curve_id]["pred"].append(pred)

    def add_meta(self, key: str, value: Union[str, float, int, bool, list]):
        """
        Add meta data to curve.

        :param key: meta key
        :param value: meta value
        """
        self._meta[key] = value

    def reset(self):
        self.__init__()

    def save(self, output_dir: str, file_name: Optional[str] = None):
        if file_name is None:
            file_name = self._generate_output_basename() + ".json"

        with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
            json.dump(OrderedDict([
                ("meta", self.meta),
                ("curves", self.curves)
            ]), f, indent=2)

    def load(self, file_name: str):
        """
        Load saved results from JSON file.
        The order of loaded curves will be preserved from the file.

        :param file_name: JSON file name
        """
        if not os.path.isfile(file_name):
            raise IOError("Input file '{}' does not exist.".format(file_name))

        with open(file_name) as f:
            json_data = json.load(f, object_pairs_hook=OrderedDict)

        if "meta" not in json_data:
            raise ValueError("No meta section")

        if "curves" not in json_data:
            raise ValueError("No curves section")

        self._meta = json_data["meta"]
        self._classes = set()
        if "classes" not in self._meta:
            for c in json_data["curves"]:
                if "cls" in c:
                    self._classes.add(c["cls"])
        else:
            self._classes = set(self._meta["classes"])

        self._curves = json_data["curves"]
        self._classes_mapping = None
        self._inv_classes_mapping = None

    def _create_numpy_label_mapping(self):
        """
        Create mapping from string labels to numpy int labels.
        """
        self._classes_mapping = {k: i for i, k in enumerate(self.meta.get("classes", []))}
        self._inv_classes_mapping = {self._classes_mapping[k]: k for k in self._classes_mapping}

    def numpy_label_to_str(self, label: int) -> Optional[str]:
        """
        Reverse mapping of numpy integer label to string label.

        :param label: numpy int label
        :return: string label for integer mapping (None if no mapping exists for `label`)
        """
        if self._classes_mapping is None:
            self._create_numpy_label_mapping()

        return self._inv_classes_mapping.get(int(label))

    def str_to_numpy_label(self, label: str) -> Optional[int]:
        """
        Mapping of string label to numpy integer label.

        :param label: string label
        :return: int label for string mapping (None if no mapping exists for `label`)
        """
        if self._classes_mapping is None:
            self._create_numpy_label_mapping()

        return self._classes_mapping.get(str(label))

    def to_numpy(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert UnmaskingResult to a numpy feature matrix containing the curve data and a numpy
        array containing the class labels.
        The feature matrix rows consist of the original curve values and their first derivative.

        String labels from the given UnmaskingResult are represented by integers (starting at 0) in the
        order in which they appear in :attr:: UnmaskingResult.meta.

        :return: numpy matrix with data samples and numpy array with integer labels (None if there are no labels)
        """
        if self._classes_mapping is None:
            self._create_numpy_label_mapping()

        curves = self.curves
        num_rows = len(curves)
        num_cols = max((len(curves[c]["values"]) for c in curves)) * 2

        # noinspection PyPep8Naming
        X = np.zeros((num_rows, num_cols))
        y = np.zeros(num_rows)

        no_labels = False
        for i, c in enumerate(curves):
            if not curves[c]["values"]:
                continue

            data = np.array(curves[c]["values"])
            X[i] = np.concatenate((data, np.gradient(data)))

            if no_labels or "cls" not in curves[c]:
                no_labels = True
            else:
                y[i] = self._classes_mapping.get(curves[c]["cls"])

        return X, (y if not no_labels else None)


class ProgressPrinter(EventHandler, Output):
    """
    Print progress events to the console.
    """

    def __init__(self, text: str = None):
        super().__init__()
        self._text = text

    @property
    def text(self) -> str:
        """Get custom display text"""
        return self._text

    @text.setter
    def text(self, text: str):
        """
        Set custom display text (overrides the native event text).
        You can use the placeholders {0}, {1} and {2} for current event number, total number
        of events and progress percentage. The usual python format string parameters are accepted.
        """
        self._text = text

    async def handle(self, name: str, event: Event, sender: type):
        """
        Accepts events:
            - ProgressEvent
        """
        if not isinstance(event, ProgressEvent):
            raise RuntimeError("event must be of type ProgressEvent")

        if self._text is None:
            print(event.text)
        else:
            total = event.events_total if event.events_total is not None else "unknown"
            percent_done = event.percent_done if event.percent_done is not None else "unknown"
            print(self._text.format(event.serial, total, percent_done))

    def save(self, output_dir: str, file_name: Optional[str] = None):
        pass

    def reset(self):
        pass


class UnmaskingStatAccumulator(EventHandler, Output):
    """
    Accumulate various statistics about a running experiment.
    """

    def __init__(self, meta_data: Optional[Dict[str, Any]] = None):
        """
        :param meta_data: dict with experiment meta data
        """
        self._initial_meta_data = meta_data if meta_data is not None else {}
        self._curves = {}
        self._meta = self._initial_meta_data
        self._classes = set()

    # noinspection PyUnresolvedReferences,PyTypeChecker
    async def handle(self, name: str, event: Event, sender: type):
        """
        Accepts events:
            - UnmaskingTrainCurveEvent
            - PairGenerationEvent
        """
        if not isinstance(event, UnmaskingTrainCurveEvent) and not isinstance(event, PairBuildingProgressEvent):
            raise TypeError("event must be of type UnmaskingTrainCurveEvent or PairBuildingProgressEvent")

        pair = event.pair
        pair_id = pair.pair_id
        if pair_id not in self._curves:
            self._curves[pair_id] = {}

        str_cls = str(pair.cls)
        self._classes.add(str_cls)

        if isinstance(event, PairBuildingProgressEvent):
            fa, fb = event.files
            self._curves[pair_id]["cls"] = str_cls
            self._curves[pair_id]["files"] = [fa, fb]
        elif isinstance(event, UnmaskingTrainCurveEvent):
            self._curves[pair_id]["values"] = event.values
            self._curves[pair_id]["fs"] = event.feature_set.__name__

    def save(self, output_dir: str, file_name: Optional[str] = None):
        """
        Save accumulated stats to file in JSON format.
        If the file exists, it will be truncated.
        """

        output = UnmaskingResult()
        for c in self._curves:
            output.add_curve(c, **self._curves[c])

        for m in self._meta:
            output.add_meta(m, self._meta[m])
        output.add_meta("classes", list(self._classes))

        if file_name is None:
            file_name = self._generate_output_basename() + ".json"
        output.save(output_dir, file_name)

    def reset(self):
        self.__init__(self._initial_meta_data)

    @property
    def meta_data(self) -> Dict[str, Any]:
        """Get experiment meta data"""
        return self._meta

    @meta_data.setter
    def meta_data(self, meta_data: Dict[str, Any]):
        """Add experiment meta data"""
        self._meta.update(meta_data)


class UnmaskingCurvePlotter(EventHandler, Output):
    """
    Plot unmasking curves.
    """
    
    def __init__(self, markers: Dict[SamplePairClass, Tuple[str, str, Optional[str]]] = None,
                 ylim: Tuple[float, float] = (0, 1.0), display: bool = False):
        """
        :param markers: dictionary of pair classes mapped to matplotlib marker codes, a
                        human-readable legend description and a color code. If color
                        is None, random colors will be chosen per curve
        :param ylim: limits of the y axis
        :param display: whether to display an interactive plot window
        """
        super().__init__()

        self._fig = None
        self._colors = {}
        self._markers = None
        self._title = ""
        if markers is not None:
            self.markers = markers
        self._display = False
        self.display = display
        self._is_being_displayed = False
        self._ylim = ylim
        self._xlim = None
        self._axes_need_update = True

        self._pyplot_styles = []
        self._output_formats = ["svg"]

        self._next_curve_id = 0
        self._curve_ids = []
        self._events_to_pair_ids = {}

        self._last_points = {}

    @property
    def styles(self) -> Union[str, dict, list]:
        """Get used pyplot styles."""
        return self._pyplot_styles

    @styles.setter
    def styles(self, styles: Union[str, dict, list]):
        """Set pyplot styles."""
        self._pyplot_styles = styles
        pyplot.style.use(styles)

    @property
    def rc_params(self) -> Dict[str, Union[str, int, float]]:
        """Get matplotlib rcParams."""
        return matplotlib.rcParams

    @rc_params.setter
    def rc_params(self, rc_params: Dict[str, Union[str, int, float]]):
        """Set matplotlib rcParams."""
        for rc in rc_params:
            matplotlib.rcParams[rc] = rc_params[rc]

    @property
    def markers(self) -> Dict[SamplePairClass, Tuple[str, str, Optional[str]]]:
        """Get markers"""
        return self._markers

    @markers.setter
    def markers(self, markers:  Dict[SamplePairClass, Tuple[str, str, Optional[str]]]):
        """Set markers"""
        self._markers = {}
        for m in markers:
            self._markers[str(m)] = markers[m]
        self._axes_need_update = True

    @property
    def ylim(self):
        """Get y axis limits"""
        return self._ylim

    @ylim.setter
    def ylim(self, ylim: Tuple[float, float]):
        """Set y axis limits"""
        self._ylim = ylim
    
    @property
    def xlim(self):
        """Get x axis limits"""
        return self._xlim
    
    @xlim.setter
    def xlim(self, xlim: Tuple[float, float]):
        """Set y axis limits"""
        self._xlim = xlim
    
    @property
    def display(self) -> bool:
        """Get whether the plot will be displayed on screen"""
        return self._display
    
    @display.setter
    def display(self, display: bool):
        """Set whether the plot will be displayed on screen"""
        if matplotlib.get_backend().lower() != "agg":
            self._display = display
        else:
            self._display = False

    @property
    def output_formats(self) -> List[str]:
        """Get output formats (default: svg)."""
        return self._output_formats

    @output_formats.setter
    def output_formats(self, ext: List[str]):
        """Set output formats."""
        self._output_formats = ext

    @property
    def title(self) -> str:
        """Plot title"""
        return self._title

    @title.setter
    def title(self, title: str):
        """Set plot title."""
        self._title = title
        self._axes_need_update = True

    async def handle(self, name: str, event: Event, sender: type):
        """
        Accepts events:
            - UnmaskingTrainCurveEvent
        """
        if not isinstance(event, UnmaskingTrainCurveEvent):
            raise TypeError("event must be of type UnmaskingTrainCurveEvent")

        if event.pair.pair_id not in self._events_to_pair_ids:
            self._events_to_pair_ids[event.pair.pair_id] = self.start_new_curve()

        self.plot_curve(event.values, event.pair.cls, self._events_to_pair_ids[event.pair.pair_id])

    def start_new_curve(self) -> int:
        """
        Start a new curve and retrieve its handle.
        
        :return: handle to the new curve, needed to append further points
        """

        if self._fig is None:
            self.reset()

        self._curve_ids.append(self._next_curve_id)
        self._last_points[self._next_curve_id] = (0, 0)
        self._next_curve_id += 1

        return self._next_curve_id - 1

    async def _flush_events_loop(self):
        """
        Helper coroutine for keeping the plot GUI responsive.
        """
        loop = asyncio.get_event_loop()
        while loop.is_running() and self._is_being_displayed and self._fig is not None:
            self._flush_events()
            await asyncio.sleep(0)

    def _flush_events(self):
        """
        Flush GUI events once.
        """
        if not self._is_being_displayed or self._fig is None:
            return

        try:
            self._fig.canvas.flush_events()
        except (AttributeError, NotImplementedError):
            pass

    def plot_curve(self, values: List[float], curve_class: Union[str, SamplePairClass], curve_handle: int):
        """
        Plot unmasking curve. Points from ``values`` which have been plotted earlier will not be plotted again.
        Consecutive calls with the same ``curve_handle`` append points new points to existing curve.
        Therefore, if you want to start a new plot for a certain curve, you need to create a new instance of
        this class or create a new figure with :method:`new_figure()`.
        
        :param values: list of y-axis values
        :param curve_class: class of the curve
        :param curve_handle: curve handle from :method:`start_new_curve()`
        """
        if self._axes_need_update:
            self._setup_axes()

        if curve_handle not in self._curve_ids:
            raise ValueError("Invalid curve ID")

        str_curve_class = str(curve_class)
        if curve_handle not in self._colors:
            if self._markers is None:
                self._markers = {}

            if str_curve_class in self._markers and self._markers[str_curve_class][2] is not None:
                self._colors[curve_handle] = self._markers[str_curve_class][2]
            else:
                self._colors[curve_handle] = "#{:02X}{:02X}{:02X}".format(
                    randint(0, 255), randint(0, 255), randint(0, 255))
                if str_curve_class not in self._markers:
                    self._markers[str_curve_class] = [".", "", None]

        num_values = len(values)
        axes = self._fig.gca()

        if num_values <= self._last_points[curve_handle][0]:
            raise ValueError("Number of curve points must be larger than for previous calls")

        if self._xlim is not None:
            axes.set_xlim(self._xlim)
        else:
            axes.set_xlim(0, max(1, max(num_values - 1, axes.get_xlim()[1])))

        marker = self._markers[str_curve_class][0]

        last_point = self._last_points[curve_handle]
        points_to_draw = values[last_point[0]:len(values)]
        xstart = last_point[0]
        last_point = (xstart, points_to_draw[0] if xstart == 0 else last_point[1])
        for i, v in enumerate(points_to_draw):
            x = (last_point[0], xstart + i)
            y = (last_point[1], v)

            axes.plot(x, y, color=self._colors[curve_handle], linestyle='solid', linewidth=1,
                      marker=marker, markersize=4)
            last_point = (x[1], y[1])
            
        self._last_points[curve_handle] = last_point

        if self._display:
            self.show()
            self._flush_events()
            self._fig.canvas.blit()
    
    def show(self):
        """Show plot area on screen."""
        if not self._is_being_displayed:
            pyplot.ion()
            self._fig.show()
            self._is_being_displayed = True
            asyncio.ensure_future(self._flush_events_loop())

    def close(self):
        """Close an open plot window."""
        pyplot.close(self._fig)
        pyplot.ioff()
        self._fig = None
        self._is_being_displayed = False
        self.reset()

    def save(self, output_dir: str, file_name: Optional[str] = None):
        if self._fig is None:
            return

        if file_name is None:
            file_name = self._generate_output_basename()
        for ext in self._output_formats:
            ext = "." + ext
            if file_name.endswith(ext):
                ext = ""
            self._fig.savefig(os.path.join(output_dir, file_name + ext))

    def reset(self):
        self._colors = {}
        self._next_curve_id = 0
        self._curve_ids = []
        self._events_to_pair_ids = {}
        self._last_points = {}
        if not self.display:
            pyplot.ioff()
        else:
            pyplot.ion()

        if self._fig is None:
            self._fig = pyplot.figure()

        self._fig.clear()
        gc.collect()
        if self._markers is not None:
            self._setup_axes()

    def _setup_axes(self):
        axes = self._fig.gca()
        axes.set_title(self._title)
        axes.set_ylim(self._ylim)
        axes.set_xlabel("rounds")
        axes.set_ylabel("accuracy")
    
        # force integer ticks on x axis
        axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        if self._ylim[0] < 0.0:
            axes.axhline(0.0, linewidth=1.0, linestyle="dashed", color="#aaaaaa")

        legend_handles = []
        legend_labels = []

        if self._markers is None:
            self._markers = {}

        for m in self._markers:
            if not self._markers[m][1]:
                continue

            if len(self._markers[m]) > 2 and self._markers[m][2] is None:
                color = "#777777"
            else:
                color = self._markers[m][2]
            legend_handles.append(pyplot.Line2D((0, 1), (0, 0), color=color, marker=self._markers[m][0]))
            legend_labels.append(self._markers[m][1])
    
        axes.legend(handles=legend_handles, labels=legend_labels)

        self._axes_need_update = False


class ModelCurvePlotter(UnmaskingCurvePlotter):
    """
    Plotter for meta model training data.
    """

    async def handle(self, name: str, event: Event, sender: type):
        """
        Accepts events:
            - ModelFitEvent
        """
        if isinstance(event, ModelFitEvent) or isinstance(event, ModelPredictEvent):
            if event.is_truth:
                self.title += " (ground truth)"
                self.title = self.title.strip()
        else:
            raise TypeError("event must be of type ModelFitEvent or ModelPredictEvent")

        labels = event.labels
        if type(labels) is not list:
            labels = list(labels)

        for i, d in enumerate(event.data):
            if type(d) is not list:
                d = list(d)
            d = d[:len(d) // 2]
            self.plot_curve(d, labels[i], self.start_new_curve())
