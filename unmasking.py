#!/usr/bin/env python3
from event import EventBroadcaster, EventHandler, ProgressEvent
from input import BookSampleParser
from classifier import SamplePair, UniqueRandomUndersampler, AvgWordFreqFeatureSet
from unmasking.strategies import FeatureRemoval


class PrintProgress(EventHandler):
    def __init__(self, text: str):
        super().__init__()
        self._text = text
        
    def handle(self, name: str, event: ProgressEvent, sender: type):
        print("{}: {:.2f}%".format(self._text, event.percent_done))


def main():
    pair_progress = PrintProgress("Pair-building progress")
    EventBroadcaster.subscribe("onProgress", pair_progress, {BookSampleParser})
    
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
        strat.run(8, 200, fs, False)

if __name__ == "__main__":
    main()
