#!/usr/bin/env python3
from event import EventBroadcaster, EventHandler, ProgressEvent
from input import BookSampleParser
from classifier import SamplePair, UniqueRandomUndersampler, AvgWordFreqFeatureSet


class PrintProgress(EventHandler):
    def __init__(self, text: str):
        super().__init__()
        self._text = text
        
    def handle(self, event: ProgressEvent, sender: type):
        print("{}: {:.2f}%".format(self._text, event.percent_done))


def main():
    chunking_progress = PrintProgress("Chunking progress")
    EventBroadcaster.subscribe("progress", chunking_progress, {BookSampleParser})
    
    parser = BookSampleParser("corpora", 500, "english")
    s = UniqueRandomUndersampler()
    
    tokenization_progress = None
    for i, pair in enumerate(parser):
        if tokenization_progress is not None:
            EventBroadcaster.unsubscribe("progress", tokenization_progress, {SamplePair})
        
        tokenization_progress = PrintProgress("Chunk tokenization progress for pair {}".format(i))
        EventBroadcaster.subscribe("progress", tokenization_progress, {SamplePair})
        
        fs = AvgWordFreqFeatureSet(pair, s)
        list(fs.get_features_absolute(20))

if __name__ == "__main__":
    main()
