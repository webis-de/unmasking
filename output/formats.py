from event.interfaces import Event, EventHandler
from event.events import ProgressEvent, UnmaskingTrainCurveEvent, PairGenerationEvent
from input.interfaces import SamplePair
from output.interfaces import FileOutput

import json
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as pyplot
from PyQt5.QtWidgets import QApplication
from random import randint
from typing import Dict, Optional, Tuple


class ProgressPrinter(EventHandler):
    """
    Print progress events to the console.
    
    Handles events: onProgress
    """
    
    def __init__(self, text: str):
        super().__init__()
        self._text = text
    
    def handle(self, name: str, event: ProgressEvent, sender: type):
        print("{}: {:.2f}%".format(self._text, event.percent_done))


class UnmaskingStatAccumulator(EventHandler, FileOutput):
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


class UnmaskingCurvePlotter(EventHandler, FileOutput):
    """
    Plot unmasking curves.
    
    Handles events: onUnmaskingRoundFinished
    """
    
    def __init__(self,  markers: Dict[SamplePair.Class, Tuple[str, str, Optional[str]]],
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
        self._drawn = {}
        self._colors = {}
        self._markers = markers
        self._display = display
        
        pyplot.ylim(ylim[0], ylim[1])
        pyplot.xlabel("rounds")
        pyplot.ylabel("discriminability")
        
        # force integer ticks on x axis
        self._fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if ylim[0] < 0.0:
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
            QApplication.processEvents()
    
    def handle(self, name: str, event: UnmaskingTrainCurveEvent, sender: type):
        if event not in self._colors:
            if self._markers[event.pair.cls][2] is not None:
                self._colors[event] = self._markers[event.pair.cls][2]
            else:
                self._colors[event] = "#{:02X}{:02X}{:02X}".format(randint(0, 255), randint(0, 255), randint(0, 255))
            self._drawn[event] = 0
        
        pyplot.xlim(0, max(pyplot.xlim()[1], event.n))
        
        points_to_draw = event.values[self._drawn[event]:len(event.values)]
        last_y = event.values[max(0, self._drawn[event] - 1)]
        last_x = max(0, len(event.values) - 2)
        
        for i, v in enumerate(points_to_draw):
            marker = self._markers[event.pair.cls][0]
            
            x = [last_x, i + self._drawn[event]]
            y = [last_y, v]
            last_x = x[1]
            last_y = y[1]
            pyplot.plot(x, y, color=self._colors[event], linestyle='solid', linewidth=1,
                        marker=marker, markersize=4)
        
        if self._display:
            QApplication.processEvents()
            self._fig.canvas.draw()
        self._drawn[event] = len(event.values)
    
    def save(self, file_name: str):
        pyplot.savefig(file_name)
