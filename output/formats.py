from event.interfaces import Event, EventHandler
from event.events import ProgressEvent, UnmaskingTrainCurveEvent, PairGenerationEvent
from input.interfaces import SamplePair
from job.interfaces import Configurable
from output.interfaces import Aggregator, FileOutput

import json
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as pyplot
from random import randint
from typing import Any, Dict, List, Optional, Tuple


class ProgressPrinter(EventHandler, Configurable):
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
    
    def handle(self, name: str, event: ProgressEvent, sender: type):
        print("{}: {:.2f}%".format(self._text, event.percent_done))


class CurveAverager(EventHandler, Aggregator, Configurable):
    """
    Average unmasking curves from multiple runs.
    
    Handles events: onUnmaskingFinished
    """

    def __init__(self):
        self._curves = {}
        self._aggregate_by_class = False

    def handle(self, name: str, event: UnmaskingTrainCurveEvent, sender: type):
        if name != "onUnmaskingFinished":
            return
        
        self.add_curve(event.pair.cls, event.pair, event.values)

    def add_curve(self, identifier: int, cls: SamplePair.Class, values: List[float]):
        agg = identifier
        if self._aggregate_by_class:
            agg = cls
            identifier = None

        if agg not in self._curves:
            self._curves[cls] = []
        self._curves[cls].append((identifier, cls, values))

    def get_aggregated_curves(self) -> Dict[Any, Tuple[int, SamplePair.Class, List[float]]]:
        avg_curves = {}
        for c in self._curves:
            curves = [c for c in zip(*self._curves[c])][2]
            avg_curves[c] = (self._curves[c][-1][0],
                             self._curves[c][-1][1],
                             [sum(e) / len(e) for e in zip(curves)])
        
        return avg_curves

    @property
    def aggregate_by_class(self):
        """ Whether to aggregate by class (default: False, i.e. aggregate by identifier) """
        return self._aggregate_by_class

    @aggregate_by_class.setter
    def aggregate_by_class(self, agg_by_class: bool):
        self._aggregate_by_class = agg_by_class


class UnmaskingStatAccumulator(EventHandler, FileOutput, Configurable):
    """
    Accumulate various statistics about a running experiment.
    
    Handles events: onPairGenerated, onUnmaskingFinished
    """
    
    def __init__(self, meta_data: Optional[Dict[str, object]] = None):
        """
        :param meta_data: dict with experiment meta data
        """
        self._stats = {}
        if meta_data is not None:
            self._stats["meta"] = meta_data
        self._stats["curves"] = {}
    
    # noinspection PyUnresolvedReferences
    def handle(self, name: str, event: Event, sender: type):
        if type(event) != UnmaskingTrainCurveEvent and type(event) != PairGenerationEvent:
            return
        
        pair = event.pair
        pair_id = id(event.pair)
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
    
    def set_meta_data(self, meta_data: Dict[str, object]):
        """
        Set experiment meta data.

        :param meta_data: meta data dict, None to unset previously set meta data
        """
        if meta_data is None and "meta" in self._stats:
            del self._stats["meta"]
        elif meta_data is not None:
            self._stats["meta"] = meta_data
    
    def save(self, file_name: str):
        """
        Save accumulated stats to file in JSON format.
        If the file exists, it will be truncated.
        """
        
        with open(file_name, "w") as f:
            json.dump(self._stats, f, indent=2)
    
    def reinit(self, meta_data: Optional[Dict[str, object]] = None):
        """
        Clear stats and reinitialize accumulator.

        :param meta_data: optional experiment meta data (None to re-use previous meta data)
        """
        if meta_data is None and "meta" in self._stats:
            meta_data = self._stats["meta"]
        
        self.__init__(meta_data)


class UnmaskingCurvePlotter(EventHandler, FileOutput, Configurable):
    """
    Plot unmasking curves.
    
    Handles events: onUnmaskingRoundFinished
    """
    
    def __init__(self,  markers: Dict[SamplePair.Class, Tuple[str, str, Optional[str]]] = None,
                 ylim: Tuple[float, float] = (0, 1.0), display: bool = True):
        """
        :param markers: dictionary of pair classes mapped to matplotlib marker codes, a
                        human-readable legend description and a color code. If color
                        is None, random colors will be chosen per curve
        :param ylim: limits of the y axis
        :param display: whether to display an interactive plot window
        """
        super().__init__()
        self._fig = pyplot.figure()
        self._colors = {}
        self._markers = None
        if markers is not None:
            self.markers = markers
        self._display = display
        self._ylim = ylim
        
        self._next_curve_id = 0
        self._curve_ids = []
        self._events_to_cids = {}
        
        self._last_points = {}
        
        if self._markers is not None:
            self._setup_axes()
        self._line = None
    
    @property
    def markers(self) -> Dict[SamplePair.Class, Tuple[str, str, Optional[str]]]:
        """Get markers"""
        return self._markers
    
    @markers.setter
    def markers(self, markers:  Dict[SamplePair.Class, Tuple[str, str, Optional[str]]]):
        """Set markers"""
        self._markers = {}
        for m in markers:
            self._markers[str(m)] = markers[m]
        self._setup_axes()
    
    @property
    def ylim(self):
        """Get y axis limits"""
        return self._ylim
    
    @property
    def display(self) -> bool:
        """Get whether the plot will be displayed on screen"""
        return self._display
    
    @display.setter
    def display(self, display: bool):
        """Set whether the plot will be displayed on screen"""
        self._display = display
    
    @ylim.setter
    def ylim(self, ylim: Tuple[float, float]):
        """Set y axis limits"""
        self._ylim = ylim
    
    def handle(self, name: str, event: UnmaskingTrainCurveEvent, sender: type):
        if event not in self._events_to_cids:
            self._events_to_cids[event] = self.start_new_curve()
        
        self.plot_curve(event.values, (0.0, event.n - 1), event.pair.cls, self._events_to_cids[event])
    
    def start_new_curve(self) -> int:
        """
        Start a new curve and retrieve its handle.
        
        :return: handle to the new curve, needed to append further points
        """
        self._curve_ids.append(self._next_curve_id)
        self._last_points[self._next_curve_id] = (0, 0)
        self._next_curve_id += 1
        return self._next_curve_id - 1
    
    def set_plot_title(self, title: str):
        """
        Set plot title.
        
        :param title: plot title
        """
        pyplot.title(title)
    
    def plot_curve(self, values: List[float], xlim: Tuple[float, float],
                   curve_class: SamplePair.Class, curve_handle: int):
        """
        Plot unmasking curve. Points from ``values`` which have been plotted earlier will not be plotted again.
        Consecutive calls with the same ``curve_handle`` append points new points to existing curve.
        Therefore, if you want to start a new plot for a certain curve, you need to create a new instance of
        this class or create a new figure with :method:`new_figure()`.
        
        :param values: list of y-axis values
        :param xlim: x-axis limits
        :param curve_class: class of the curve
        :param curve_handle: curve handle from :method:`start_new_curve()`
        """
        if curve_handle not in self._curve_ids:
            raise ValueError("Invalid curve ID")
        
        if curve_handle not in self._colors:
            if self._markers[str(curve_class)][2] is not None:
                self._colors[curve_handle] = self._markers[str(curve_class)][2]
            else:
                self._colors[curve_handle] = "#{:02X}{:02X}{:02X}".format(randint(0, 255), randint(0, 255), randint(0, 255))
        
        if len(values) <= self._last_points[curve_handle][0]:
            raise ValueError("Number of curve points must be larger than for previous calls")
        
        pyplot.xlim(xlim[0], max(pyplot.xlim()[1], xlim[1]))
        marker = self._markers[str(curve_class)][0]
        
        last_point = self._last_points[curve_handle]
        points_to_draw = values[last_point[0]:len(values)]
        xstart = last_point[0]
        last_point = (xstart, points_to_draw[0] if xstart == 0 else last_point[1])
        for i, v in enumerate(points_to_draw):
            x = (last_point[0], xstart + i)
            y = (last_point[1], v)
            
            pyplot.plot(x, y, color=self._colors[curve_handle], linestyle='solid', linewidth=1,
                        marker=marker, markersize=4)
            last_point = (x[1], y[1])
            
        self._last_points[curve_handle] = last_point
        
        if self._display:
            self._fig.canvas.blit()
            self._fig.canvas.flush_events()
    
    def new_figure(self):
        """Create a new figure and reset record of already drawn curves."""
        self._fig.clear()
        self._fig = pyplot.figure()
        self._setup_axes()
        self._curve_ids = []
        self._next_curve_id = 0
    
    def close(self):
        """Close an open plot window ."""
        pyplot.close()
    
    def save(self, file_name: str):
        pyplot.savefig(file_name)
    
    def _setup_axes(self):
        pyplot.ylim(self._ylim[0], self._ylim[1])
        pyplot.xlabel("rounds")
        pyplot.ylabel("discriminability")
    
        # force integer ticks on x axis
        self._fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
        if self._ylim[0] < 0.0:
            pyplot.axhline(0, linewidth=1.0, linestyle="dashed", color="#aaaaaa")
    
        legend_handles = []
        legend_labels = []
        for m in self._markers:
            if len(self._markers[m]) > 2 and self._markers[m][2] is None:
                color = "#777777"
            else:
                color = self._markers[m][2]
            legend_handles.append(pyplot.Line2D((0, 1), (0, 0), color=color, marker=self._markers[m][0]))
            legend_labels.append(self._markers[m][1])
    
        pyplot.legend(handles=legend_handles, labels=legend_labels)
    
        if self._display:
            pyplot.ion()
            pyplot.show(block=False)
            self._fig.canvas.flush_events()
