#!/usr/bin/env python3
from event.interfaces import Event, EventHandler
from event.dispatch import EventBroadcaster
from event.events import ProgressEvent, UnmaskingTrainCurveEvent, PairGenerationEvent
from input.interfaces import SamplePair
from input.formats import BookSampleParser, WebisBuzzfeedCatCorpusParser, WebisBuzzfeedAuthorshipCorpusParser
from classifier.features import AvgWordFreqFeatureSet, AvgCharNgramFreqFeatureSet, AvgDisjunctCharNgramFreqFeatureSet
from classifier.sampling import UniqueRandomUndersampler
from input.tokenizers import SentenceChunkTokenizer, PassthroughTokenizer
from unmasking.strategies import FeatureRemoval

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as pyplot
from PyQt5.QtWidgets import QApplication

import json
import os
from random import randint
from time import time
from typing import Dict, Optional, Tuple


class PrintProgress(EventHandler):
    def __init__(self, text: str):
        super().__init__()
        self._text = text
        
    def handle(self, name: str, event: ProgressEvent, sender: type):
        print("{}: {:.2f}%".format(self._text, event.percent_done))


class UnmaskingStatAccumulator(EventHandler):
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
            self._stats["curves"][pair_id]["curve"] = [max(0, (v - .5) * 2) for v in event.values]
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
    
    def save(self, file_name: str, append: bool = True):
        """
        Save accumulated stats to file in JSON format.
        If the file exists, it will be truncated.
        
        :param file_name: output file name
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
        QApplication.processEvents()
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
        
        corpus            = "buzzfeed"
        removed_per_round = 10
        iterations        = 25
        num_features      = 250

        stats_accumulator = UnmaskingStatAccumulator()
        EventBroadcaster.subscribe("onPairGenerated", stats_accumulator)
        EventBroadcaster.subscribe("onUnmaskingFinished", stats_accumulator)
        
        if corpus == "buzzfeed":
            experiment = "orientation"
            
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
                #fs = AvgDisjunctCharNgramFreqFeatureSet(pair, s, 3)
                strat = FeatureRemoval(removed_per_round)
                strat.run(iterations, num_features, fs, False)
            
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
                strat = FeatureRemoval(removed_per_round)
                strat.run(iterations, num_features, fs, False)
        else:
            raise ValueError("Invalid corpus")
        
        # save stats
        stats_accumulator.set_meta_data({
            "removed_per_round": removed_per_round,
            "iterations": iterations,
            "num_features": num_features,
            "chunk_tokenizer": chunk_tokenizer.__class__.__name__
        })
        output_filename = "out/unmasking_" + str(int(time()))
        print("Saving plot to '{}.svg'...".format(output_filename))
        pyplot.savefig(output_filename + ".svg")
        print("Writing experiment meta data to '{}.json'...".format(output_filename))
        if not os.path.exists("out"):
            os.mkdir("out")
        stats_accumulator.save(output_filename + ".json")
        
        print("Time taken: {:.03f} seconds.".format(time() - start_time))

        # block, so window doesn't close automatically
        pyplot.show(block=True)
    except KeyboardInterrupt:
        print("Exited upon user request.")
        exit(1)

if __name__ == "__main__":
    main()
