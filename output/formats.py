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
from util.util import get_base_path

from random import randint
from typing import Any, Dict

import asyncio
import json
import matplotlib
import matplotlib.ticker
import os
import sys
import yaml

# don't use default Qt backend if we are operating without a display server
if sys.platform == "linux" and os.environ.get("DISPLAY") is None:
    matplotlib.use("Agg")

try:
    import matplotlib.pyplot as pyplot
except (ModuleNotFoundError, ImportError):
    matplotlib.use("Agg", warn=False, force=True)
    import matplotlib.pyplot as pyplot


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

        self._stats = {
            "meta": self._initial_meta_data,
            "curves": {}
        }

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
        if event.pair is not None and pair_id not in self._stats["curves"]:
            self._stats["curves"][pair_id] = {}

        str_cls = str(pair.cls)
        self._classes.add(str_cls)

        if isinstance(event, PairBuildingProgressEvent):
            fa, fb = event.files
            self._stats["curves"][pair_id]["cls"] = str_cls
            self._stats["curves"][pair_id]["files_a"] = fa
            self._stats["curves"][pair_id]["files_b"] = fb
        elif isinstance(event, UnmaskingTrainCurveEvent):
            self._stats["curves"][pair_id]["curve"] = event.values
            self._stats["curves"][pair_id]["fs"] = event.feature_set.__name__
    
    def save(self, output_dir: str, file_name: Optional[str] = None):
        """
        Save accumulated stats to file in JSON format.
        If the file exists, it will be truncated.
        """

        if file_name is None:
            file_name = os.path.join(output_dir, self._generate_output_basename() + ".json")

        self._stats["meta"]["classes"] = sorted(self._classes)
        with open(file_name, "w") as f:
            json.dump(self._stats, f, indent=2)

    def reset(self):
        self.__init__(self._initial_meta_data)

    @property
    def meta_data(self) -> Dict[str, Any]:
        """Get experiment meta data"""
        if "meta" in self._stats:
            return self._stats["meta"]
        return {}
    
    @meta_data.setter
    def meta_data(self, meta_data: Dict[str, Any]):
        """Set experiment meta data"""
        self._stats["meta"] = meta_data


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

        self._rc_file = None
        self._fig = None
        self._colors = {}
        self._markers = None
        if markers is not None:
            self.markers = markers
        self._display = False
        self._is_being_displayed = False
        self._ylim = ylim
        self._xlim = None
        self._axes_need_update = True

        self.display = display
        
        self._next_curve_id = 0
        self._curve_ids = []
        self._events_to_pair_ids = {}
        
        self._last_points = {}

    @property
    def rc_file(self) -> str:
        """Get plot RC file"""
        return self._rc_file

    @rc_file.setter
    def rc_file(self, rc_file: str):
        """Set and parse plot RC file"""
        rc_file = os.path.join(get_base_path(), rc_file)
        self._rc_file = rc_file
        with open(rc_file, "r") as f:
            rc_contents = yaml.safe_load(f)

        self.markers = rc_contents.get("markers", {})
        pyplot.style.use(rc_contents.get("styles", []))
        rc_params = rc_contents.get("rc_params", {})
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

    def set_plot_title(self, title: str):
        """
        Set plot title.
        
        :param title: plot title
        """
        self._fig.gca().set_title(title)

    async def _flush_events_loop(self):
        """
        Helper coroutine for keeping the plot GUI responsive.
        """
        loop = asyncio.get_event_loop()
        while loop.is_running() and self._is_being_displayed and self._fig is not None:
            self._flush_events()
            await asyncio.sleep(0.0001)

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

    def plot_curve(self, values: List[float], curve_class: SamplePairClass, curve_handle: int):
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
        self._fig = None
        self._is_being_displayed = False
        self.reset()

    def save(self, output_dir: str, file_name: Optional[str] = None):
        if self._fig is None:
            return

        if file_name is None:
            file_name = self._generate_output_basename() + ".svg"
        self._fig.savefig(os.path.join(output_dir, file_name))

    def reset(self):
        self._colors = {}
        self._next_curve_id = 0
        self._curve_ids = []
        self._events_to_pair_ids = {}
        self._last_points = {}

        if self._fig is None:
            self._fig = pyplot.figure()

        self._fig.clear()
        if self._markers is not None:
            self._setup_axes()

    def _setup_axes(self):
        axes = self._fig.gca()
        axes.set_ylim(self._ylim)
        axes.set_xlabel("rounds")
        axes.set_ylabel("accuracy")
    
        # force integer ticks on x axis
        axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        if self._ylim[0] < 0.0:
            axes.axhline(0.0, linewidth=1.0, linestyle="dashed", color="#aaaaaa")

        legend_handles = []
        legend_labels = []
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
