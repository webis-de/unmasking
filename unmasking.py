#!/usr/bin/env python3
from event import EventBroadcaster, EventHandler, ProgressEvent, UnmaskingTrainCurveEvent
from input import BookSampleParser
from classifier import SamplePair, UniqueRandomUndersampler, AvgWordFreqFeatureSet
from unmasking.strategies import FeatureRemoval

import pylab

from random import randint


class PrintProgress(EventHandler):
    def __init__(self, text: str):
        super().__init__()
        self._text = text
        
    def handle(self, name: str, event: ProgressEvent, sender: type):
        print("{}: {:.2f}%".format(self._text, event.percent_done))


class PlotUnmaskingCurve(EventHandler):
    def __init__(self):
        super().__init__()
        self._fig = pylab.figure()
        self._drawn = {}
        self._colors = {}
        pylab.ion()
        pylab.ylim(0, 1.0)
        pylab.xlabel("rounds")
        pylab.ylabel("accuracy")

        same_l      = pylab.Line2D((0, 1), (0, 0), color='#777777', marker='o')
        different_l = pylab.Line2D((0, 1), (0, 0), color='#777777', marker='x')
        pylab.legend(handles=(same_l, different_l), labels=("same author", "different authors"))
    
    def handle(self, name: str, event: UnmaskingTrainCurveEvent, sender: type):
        if event not in self._colors:
            self._colors[event] = "#{:02X}{:02X}{:02X}".format(randint(0, 255), randint(0, 255), randint(0, 255))
            self._drawn[event] = 0
        
        pylab.xlim(0, max(pylab.xlim()[1], event.n))
        points_to_draw = event.values[self._drawn[event]:len(event.values)]
        for i, v in enumerate(points_to_draw):
            if event.pair.cls == SamplePair.Class.SAME_AUTHOR:
                marker = "o"
            else:
                marker = "x"
            
            pylab.plot(i + self._drawn[event], v, color=self._colors[event], linestyle='solid', marker=marker, markersize=4)

        self._drawn[event] = len(event.values)
        self._fig.canvas.draw()
        pylab.show(block=False)
        

def main():
    pair_progress = PrintProgress("Pair-building progress")
    EventBroadcaster.subscribe("onProgress", pair_progress, {BookSampleParser})
    EventBroadcaster.subscribe("onUnmaskingRoundFinished", PlotUnmaskingCurve())
    
    parser = BookSampleParser("corpora", 500, "english")
    s = UniqueRandomUndersampler()

    chunking_progress = None
    for i, pair in enumerate(parser):
        if chunking_progress is not None:
            EventBroadcaster.unsubscribe("onProgress", chunking_progress, {SamplePair})
        
        chunking_progress = PrintProgress("Chunking progress for pair {}".format(i))
        EventBroadcaster.subscribe("onProgress", chunking_progress, {SamplePair})
        
        fs = AvgWordFreqFeatureSet(pair, s)
        strat = FeatureRemoval(8)
        strat.run(10, 250, fs, False)
    
    pylab.show(block=True)

if __name__ == "__main__":
    main()
