#!/usr/bin/env python3
from event import EventBroadcaster, EventHandler, ProgressEvent, UnmaskingTrainCurveEvent
from input import BookSampleParser
from classifier import SamplePair, UniqueRandomUndersampler, AvgWordFreqFeatureSet
from unmasking.strategies import FeatureRemoval

import matplotlib.pyplot as pyplot

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
        self._fig = pyplot.figure()
        self._drawn = {}
        self._colors = {}
        pyplot.ion()
        pyplot.ylim(0, 1.0)
        pyplot.xlabel("rounds")
        pyplot.ylabel("discriminability")

        leg1 = pyplot.Line2D((0, 1), (0, 0), color='#777777', marker='o')
        leg2 = pyplot.Line2D((0, 1), (0, 0), color='#777777', marker='x')
        pyplot.legend(handles=(leg1, leg2), labels=("same author", "different authors"))
    
    def handle(self, name: str, event: UnmaskingTrainCurveEvent, sender: type):
        if event not in self._colors:
            self._colors[event] = "#{:02X}{:02X}{:02X}".format(randint(0, 255), randint(0, 255), randint(0, 255))
            self._drawn[event] = 0

        pyplot.xlim(0, max(pyplot.xlim()[1], event.n))
        pyplot.xticks(range(0, int(pyplot.xlim()[1])))
        points_to_draw = event.values[self._drawn[event]:len(event.values)]
        last_y = event.values[max(0, self._drawn[event] - 1)]
        last_x = max(0, len(event.values) - 2)
        for i, v in enumerate(points_to_draw):
            if event.pair.cls == SamplePair.Class.SAME_AUTHOR:
                marker = "o"
            else:
                marker = "x"
            
            x = [last_x, i + self._drawn[event]]
            y = [last_y, v]
            last_x = x[1]
            last_y = y[1]
            pyplot.plot(x, y, color=self._colors[event], linestyle='solid', linewidth=1,
                        marker=marker, markersize=4)

        self._drawn[event] = len(event.values)
        self._fig.canvas.draw()
        pyplot.show(block=False)
        

def main():
    try:
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
            strat = FeatureRemoval(10)
            strat.run(20, 250, fs, False)
    
        # block, so window doesn't close automatically
        pyplot.show(block=True)
    except KeyboardInterrupt:
        print("Exited upon user request.")
        exit(1)

if __name__ == "__main__":
    main()
