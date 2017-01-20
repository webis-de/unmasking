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
    parser = BookSampleParser("corpora", 500, "english")
    s = UniqueRandomUndersampler()
    
    chunking_progress = PrintProgress("Chunking progress")
    EventBroadcaster.subscribe("progress", chunking_progress, {parser.__class__})
    
    tokenization_progress = PrintProgress("Chunk tokenization progress")
    EventBroadcaster.subscribe("progress", tokenization_progress, {SamplePair})
    
    for pair in parser:
        fs = AvgWordFreqFeatureSet(pair, s)
        list(fs.get_features_absolute(20))

if __name__ == "__main__":
    main()
