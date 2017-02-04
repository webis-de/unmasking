#!/usr/bin/env python3
from event import EventBroadcaster, EventHandler, ProgressEvent, UnmaskingTrainCurveEvent
from input import BookSampleParser, WebisBuzzfeedCatCorpusParser, WebisBuzzfeedAuthorshipCorpusParser, SamplePair
from classifier import UniqueRandomUndersampler, AvgWordFreqFeatureSet, AvgCharNgramFreqFeatureSet
from input.tokenizers import SentenceChunkTokenizer, PassthroughTokenizer
from unmasking.strategies import FeatureRemoval

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as pyplot
from PyQt5 import QtWidgets

from random import randint
from time import time
from typing import Dict, Optional, Tuple


class PrintProgress(EventHandler):
    def __init__(self, text: str):
        super().__init__()
        self._text = text
        
    def handle(self, name: str, event: ProgressEvent, sender: type):
        print("{}: {:.2f}%".format(self._text, event.percent_done))


class UnmaskingCurvePlotter(EventHandler):
    def __init__(self, markers: Dict[SamplePair.Class, Tuple[str, str, Optional[str]]],
                 ylim : Tuple[float, float] = (0, 1.0), cap_bottom: bool = True):
        """
        :param markers: dictionary of pair classes mapped to matplotlib marker codes, a
                        human-readable legend description and a color code. If color
                        is None, random colors will be chosen per curve
        :param ylim: limits of the y axis
        :param cap_bottom: whether to cap normalized y values at 0
        """
        super().__init__()
        self._fig = pyplot.figure()
        self._drawn = {}
        self._colors = {}
        self._markers = markers
        self._cap_bottom = cap_bottom
        
        pyplot.ylim(ylim[0], ylim[1])
        pyplot.xlabel("rounds")
        pyplot.ylabel("discriminability")
        
        # force integer ticks on x axis
        self._fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if ylim[0] < 0.0:
            pyplot.axhline(0, linewidth=1.0, linestyle="dashed", color="#aaaaaa")
        
        legend_handles = []
        legend_labels  = []
        for m in self._markers:
            if len(self._markers[m]) > 2 and self._markers[m][2] is None:
                color = "#777777"
            else:
                color = self._markers[m][2]
            legend_handles.append(pyplot.Line2D((0, 1), (0, 0), color=color, marker=self._markers[m][0]))
            legend_labels.append(self._markers[m][1])
        
        pyplot.legend(handles=legend_handles, labels=legend_labels)
        pyplot.ion()
        pyplot.show(block=False)
        QtWidgets.QApplication.processEvents()
    
    def handle(self, name: str, event: UnmaskingTrainCurveEvent, sender: type):
        if event not in self._colors:
            if self._markers[event.pair.cls][2] is not None:
                self._colors[event] = self._markers[event.pair.cls][2]
            else:
                self._colors[event] = "#{:02X}{:02X}{:02X}".format(randint(0, 255), randint(0, 255), randint(0, 255))
            self._drawn[event] = 0

        pyplot.xlim(0, max(pyplot.xlim()[1], event.n))
        
        points_to_draw = event.values[self._drawn[event]:len(event.values)]
        last_y = self._normalize(event.values[max(0, self._drawn[event] - 1)])
        last_x = max(0, len(event.values) - 2)
        
        for i, v in enumerate(points_to_draw):
            # normalize v
            v = self._normalize(v)
            
            marker = self._markers[event.pair.cls][0]
            
            x = [last_x, i + self._drawn[event]]
            y = [last_y, v]
            last_x = x[1]
            last_y = y[1]
            pyplot.plot(x, y, color=self._colors[event], linestyle='solid', linewidth=1,
                        marker=marker, markersize=4)
        QtWidgets.QApplication.processEvents()
        self._fig.canvas.draw()
        self._drawn[event] = len(event.values)
    
    def _normalize(self, val: float):
        if self._cap_bottom:
            return max(0, (val - .5) * 2.0)
        return (val - .5) * 2.0
        

def main():
    try:
        start_time = time()
        
        pair_progress = PrintProgress("Pair-building progress")
        EventBroadcaster.subscribe("onProgress", pair_progress, {BookSampleParser})
        
        corpus = "buzzfeed"

        if corpus == "buzzfeed":
            experiment = "portal_authorship"
            
            chunk_tokenizer = PassthroughTokenizer()
            
            if experiment == "orientation":
                labels = {
                    WebisBuzzfeedCatCorpusParser.PairClass.LEFT_LEFT: ("<", "left-left", "#990000"),
                    WebisBuzzfeedCatCorpusParser.PairClass.RIGHT_RIGHT: (">", "right-right", "#eeaa00"),
                    WebisBuzzfeedCatCorpusParser.PairClass.MAINSTREAM_MAINSTREAM: ("^", "mainstream-mainstream", "#009900"),
                    WebisBuzzfeedCatCorpusParser.PairClass.LEFT_RIGHT: ("x", "left-right", "#000099"),
                    WebisBuzzfeedCatCorpusParser.PairClass.LEFT_MAINSTREAM: ("v", "left-mainstream", "#009999"),
                    WebisBuzzfeedCatCorpusParser.PairClass.RIGHT_MAINSTREAM: ("D", "right-mainstream", "#aa6600")
                }

                parser = WebisBuzzfeedCatCorpusParser("corpora/buzzfeed", chunk_tokenizer,
                                                      ["articles_buzzfeed1", "articles_buzzfeed2"],
                                                      WebisBuzzfeedCatCorpusParser.class_by_orientation)
            elif experiment == "veracity":
                labels = {
                    WebisBuzzfeedCatCorpusParser.PairClass.FAKE_FAKE: ("<", "fake-fake", "#990000"),
                    WebisBuzzfeedCatCorpusParser.PairClass.REAL_REAL: (">", "real-real", "#eeaa00"),
                    WebisBuzzfeedCatCorpusParser.PairClass.FAKE_REAL: ("x", "fake-real", "#000099"),
                    WebisBuzzfeedCatCorpusParser.PairClass.FAKE_SATIRE: ("v", "fake-satire", "#009999"),
                    WebisBuzzfeedCatCorpusParser.PairClass.SATIRE_REAL: ("D", "satire-real", "#aa6600")
                }

                parser = WebisBuzzfeedCatCorpusParser("corpora/buzzfeed", chunk_tokenizer,
                                                      ["articles_buzzfeed1", "articles_buzzfeed2"],
                                                      WebisBuzzfeedCatCorpusParser.class_by_veracity)
            elif experiment == "portal_authorship":
                labels = {
                    WebisBuzzfeedAuthorshipCorpusParser.Class.SAME_PORTAL: ("o", "same portal", None),
                    WebisBuzzfeedAuthorshipCorpusParser.Class.DIFFERENT_PORTALS: ("x", "different portals", None)
                }
                parser = WebisBuzzfeedAuthorshipCorpusParser("corpora/buzzfeed", chunk_tokenizer,
                                                             ["articles_buzzfeed1"])
            else:
                raise ValueError("Invalid experiment")
            
            EventBroadcaster.subscribe("onUnmaskingRoundFinished", UnmaskingCurvePlotter(labels, (-.2, 1.0), True))
            s = UniqueRandomUndersampler()
            for i, pair in enumerate(parser):
                fs = AvgWordFreqFeatureSet(pair, s)
                #fs = AvgCharNgramFreqFeatureSet(pair, s, 3)
                strat = FeatureRemoval(10)
                strat.run(30, 250, fs, False)
            
        elif corpus == "gutenberg_test":
            EventBroadcaster.subscribe("onUnmaskingRoundFinished", UnmaskingCurvePlotter({
                BookSampleParser.Class.SAME_AUTHOR: ("o", "same author", None),
                BookSampleParser.Class.DIFFERENT_AUTHORS: ("x", "different authors", None)
            }))
            
            chunk_tokenizer = SentenceChunkTokenizer(500)
            parser = BookSampleParser("corpora/gutenberg_test", chunk_tokenizer)
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
        else:
            raise ValueError("Invalid corpus")

        print("Time taken: {:.03f} seconds.".format(time() - start_time))

        # block, so window doesn't close automatically
        pyplot.show()
    except KeyboardInterrupt:
        print("Exited upon user request.")
        exit(1)

if __name__ == "__main__":
    main()
