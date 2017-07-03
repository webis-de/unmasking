import json
import os
from random import randint
from typing import Any, Dict

import asyncio
import matplotlib.pyplot as pyplot

from matplotlib.ticker import MaxNLocator
from event.events import *
from event.interfaces import EventHandler
from input.interfaces import SamplePairClass
from output.interfaces import Output


class ProgressPrinter(EventHandler, Output):
    """
    Print progress events to the console.
    
    Handles events: onProgress
    """

    def __init__(self, text: str = None):
        super().__init__()
        self._text = text
    
    @property
    def text(self) -> str:
        """Get display text"""
        return self._text
    
    @text.setter
    def text(self, text: str):
        """Set display text"""
        self._text = text
    
    async def handle(self, name: str, event: ProgressEvent, sender: type):
        print("{}: {:.2f}%".format(self._text, event.percent_done))

    def save(self, output_dir: str):
        pass

    def reset(self):
        pass


class UnmaskingStatAccumulator(EventHandler, Output):
    """
    Accumulate various statistics about a running experiment.
    
    Handles events: onPairGenerated, onUnmaskingFinished
    """

    def __init__(self, meta_data: Optional[Dict[str, Any]] = None):
        """
        :param meta_data: dict with experiment meta data
        """
        self._stats = {
            "meta": meta_data if meta_data is not None else {},
            "curves": {}
        }

    # noinspection PyUnresolvedReferences,PyTypeChecker
    async def handle(self, name: str, event: Event, sender: type):
        if isinstance(event, UnmaskingTrainCurveEvent) and isinstance(event, PairGenerationEvent):
            raise TypeError("event must be of type UnmaskingTrainCurveEvent or PairGenerationEvent")
        
        pair = event.pair
        pair_id = pair.pair_id
        if event.pair is not None and pair_id not in self._stats["curves"]:
            self._stats["curves"][pair_id] = {}
        
        if name == "onPairGenerated":
            fa, fb = event.files
            self._stats["curves"][pair_id]["cls"] = str(pair.cls)
            self._stats["curves"][pair_id]["files_a"] = fa
            self._stats["curves"][pair_id]["files_b"] = fb
        elif name == "onUnmaskingFinished":
            self._stats["curves"][pair_id]["curve"] = event.values
            self._stats["curves"][pair_id]["fs"] = event.feature_set.__name__
    
    def save(self, output_dir: str):
        """
        Save accumulated stats to file in JSON format.
        If the file exists, it will be truncated.
        """

        file_name = os.path.join(output_dir, self._get_output_filename_base() + ".json")
        with open(file_name, "w") as f:
            json.dump(self._stats, f, indent=2)

    def reset(self):
        meta_data = self._stats["meta"]
        self.__init__(meta_data)
    
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
    
    Handles events: onUnmaskingRoundFinished
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
        if markers is not None:
            self.markers = markers
        self._display = display
        self._is_being_displayed = False
        self._ylim = ylim
        self._xlim = None
        self._axes_need_update = True
        
        self._next_curve_id = 0
        self._curve_ids = []
        self._events_to_pair_ids = {}
        
        self._last_points = {}
    
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
        self._display = display

    async def handle(self, name: str, event: Event, sender: type):
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

    async def _flush_gui_events(self):
        """
        Helper coroutine for keeping the plot GUI responsive.
        """
        loop = asyncio.get_event_loop()
        while loop.is_running() and self._is_being_displayed and self._fig is not None:
            self._fig.canvas.flush_events()
            await asyncio.sleep(0)
    
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
        
        if curve_handle not in self._colors:
            if self._markers[str(curve_class)][2] is not None:
                self._colors[curve_handle] = self._markers[str(curve_class)][2]
            else:
                self._colors[curve_handle] = "#{:02X}{:02X}{:02X}".format(randint(0, 255), randint(0, 255), randint(0, 255))

        num_values = len(values)
        axes = self._fig.gca()

        if num_values <= self._last_points[curve_handle][0]:
            raise ValueError("Number of curve points must be larger than for previous calls")

        if self._xlim is not None:
            axes.set_xlim(self._xlim)
        else:
            axes.set_xlim(0, max(1, max(num_values - 1, axes.get_xlim()[1])))

        marker = self._markers[str(curve_class)][0]

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
            self._fig.canvas.flush_events()
            self._fig.canvas.blit()
    
    def show(self):
        """Show plot area on screen."""
        if not self._is_being_displayed:
            pyplot.ion()
            self._fig.show()
            self._is_being_displayed = True
            asyncio.ensure_future(self._flush_gui_events())
    
    def close(self):
        """Close an open plot window."""
        pyplot.close(self._fig)
        self._fig = None
        self.reset()
    
    def save(self, output_dir: str):
        if self._fig is not None:
            self._fig.savefig(os.path.join(output_dir, self._get_output_filename_base() + ".svg"))

    def reset(self):
        self._next_curve_id = 0
        self._curve_ids = []
        self._events_to_pair_ids = {}
        self._last_points = {}
        self._is_being_displayed = False

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
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    
        if self._ylim[0] < 0.0:
            axes.axhline(0.0, linewidth=1.0, linestyle="dashed", color="#aaaaaa")
    
        legend_handles = []
        legend_labels = []
        for m in self._markers:
            if len(self._markers[m]) > 2 and self._markers[m][2] is None:
                color = "#777777"
            else:
                color = self._markers[m][2]
            legend_handles.append(pyplot.Line2D((0, 1), (0, 0), color=color, marker=self._markers[m][0]))
            legend_labels.append(self._markers[m][1])
    
        axes.legend(handles=legend_handles, labels=legend_labels)

        self._axes_need_update = False
