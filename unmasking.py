#!/usr/bin/env python3
from event import EventBroadcaster, EventHandler, ProgressEvent, UnmaskingTrainCurveEvent
from input import BookSampleParser, BuzzFeedXMLCorpusParser, SamplePair
from classifier import UniqueRandomUndersampler, AvgWordFreqFeatureSet
from input.tokenizers import SentenceChunkTokenizer, PassthroughTokenizer
from unmasking.strategies import FeatureRemoval

import matplotlib.pyplot as pyplot

from random import randint
from time import time
from typing import Dict, Tuple


class PrintProgress(EventHandler):
    def __init__(self, text: str):
        super().__init__()
        self._text = text
        
    def handle(self, name: str, event: ProgressEvent, sender: type):
        print("{}: {:.2f}%".format(self._text, event.percent_done))


class PlotUnmaskingCurve(EventHandler):
    def __init__(self, markers: Dict[SamplePair.Class, Tuple[str, str]]):
        """
        :param markers: dictionary of pair classes mapped to matplotlib marker codes and a
                        human-readable legend description
        """
        super().__init__()
        self._fig = pyplot.figure()
        self._drawn = {}
        self._colors = {}
        self._markers = markers
        pyplot.ion()
        pyplot.ylim(0, 1.0)
        pyplot.xlabel("rounds")
        pyplot.ylabel("discriminability")

        legend_handles = []
        legend_labels  = []
        for m in self._markers:
            legend_handles.append(pyplot.Line2D((0, 1), (0, 0), color='#777777', marker=self._markers[m][0]))
            legend_labels.append(self._markers[m][1])
        
        pyplot.legend(handles=legend_handles, labels=legend_labels)
    
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
            marker = self._markers[event.pair.cls][0]
            
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
        start_time = time()
        
        pair_progress = PrintProgress("Pair-building progress")
        EventBroadcaster.subscribe("onProgress", pair_progress, {BookSampleParser})
        
        corpus = "buzzfeed"

        if corpus == "buzzfeed":
            EventBroadcaster.subscribe("onUnmaskingRoundFinished", PlotUnmaskingCurve({
                BuzzFeedXMLCorpusParser.PairClass.LEFT_LEFT: ("<", "left-left"),
                BuzzFeedXMLCorpusParser.PairClass.RIGHT_RIGHT: (">", "right-right"),
                BuzzFeedXMLCorpusParser.PairClass.MAINSTREAM_MAINSTREAM: ("^", "mainstream-mainstream"),
                BuzzFeedXMLCorpusParser.PairClass.LEFT_RIGHT: ("x", "left-right"),
                BuzzFeedXMLCorpusParser.PairClass.LEFT_MAINSTREAM: ("v", "left-mainstream"),
                BuzzFeedXMLCorpusParser.PairClass.RIGHT_MAINSTREAM: ("D", "right-mainstream")
            }))
            
            chunk_tokenizer = PassthroughTokenizer()
            parser = BuzzFeedXMLCorpusParser("corpora/buzzfeed", chunk_tokenizer, ["articles_buzzfeed1", "articles_buzzfeed2"],
                                             BuzzFeedXMLCorpusParser.class_by_orientation)
            s = UniqueRandomUndersampler()
            for i, pair in enumerate(parser):
                fs = AvgWordFreqFeatureSet(pair, s)
                strat = FeatureRemoval(10)
                strat.run(10, 250, fs, False)
            
        elif corpus == "gutenberg_test":
            EventBroadcaster.subscribe("onUnmaskingRoundFinished", PlotUnmaskingCurve({
                BookSampleParser.Class.SAME_AUTHOR: ("o", "same author"),
                BookSampleParser.Class.DIFFERENT_AUTHORS: ("x", "different authors")
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

        print("Time taken: {:.03f} seconds.".format(time() - start_time))

        # block, so window doesn't close automatically
        pyplot.show(block=True)
    except KeyboardInterrupt:
        print("Exited upon user request.")
        exit(1)

if __name__ == "__main__":
    main()
