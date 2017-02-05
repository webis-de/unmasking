#!/usr/bin/env python3
from classifier.features import AvgWordFreqFeatureSet, AvgCharNgramFreqFeatureSet, AvgDisjunctCharNgramFreqFeatureSet
from classifier.sampling import UniqueRandomUndersampler
from event.dispatch import EventBroadcaster
from input.interfaces import SamplePair
from input.formats import BookSampleParser, WebisBuzzfeedCatCorpusParser, WebisBuzzfeedAuthorshipCorpusParser
from input.tokenizers import SentenceChunkTokenizer, PassthroughTokenizer
from output.formats import ProgressPrinter, UnmaskingStatAccumulator, UnmaskingCurvePlotter, CurveAverager
from unmasking.strategies import FeatureRemoval
from util.cache import CacheMixin

import os
from time import time


def save_output(curve_plotter, stats_accumulator):
    output_filename = "out/unmasking_" + str(int(time()))
    
    print("Saving plot to '{}.svg'...".format(output_filename))
    curve_plotter.save(output_filename + ".svg")
    
    print("Writing experiment meta data to '{}.json'...".format(output_filename))
    if not os.path.exists("out"):
        os.mkdir("out")
    stats_accumulator.save(output_filename + ".json")


def main():
    try:
        start_time = time()
        
        pair_progress = ProgressPrinter("Pair-building progress")
        EventBroadcaster.subscribe("onProgress", pair_progress, {BookSampleParser})
        
        corpus            = "buzzfeed"
        removed_per_round = 10
        iterations        = 25
        num_features      = 250
        num_experiments   = 5

        stats_accumulator = UnmaskingStatAccumulator()
        EventBroadcaster.subscribe("onPairGenerated", stats_accumulator)
        EventBroadcaster.subscribe("onUnmaskingFinished", stats_accumulator)
        
        if corpus == "buzzfeed":
            experiment = "orientation"
            
            chunk_tokenizer = PassthroughTokenizer()

            curve_averager = CurveAverager()
            EventBroadcaster.subscribe("onUnmaskingFinished", curve_averager)
            
            for experiment_num in range(0, num_experiments):
                CacheMixin.reset_caches()
                
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
                        #WebisBuzzfeedCatCorpusParser.PairClass.FAKE_FAKE: ("<", "fake-fake", "#990000"),
                        #WebisBuzzfeedCatCorpusParser.PairClass.REAL_REAL: (">", "real-real", "#eeaa00"),
                        #WebisBuzzfeedCatCorpusParser.PairClass.FAKE_REAL: ("x", "fake-real", "#000099"),
                        #WebisBuzzfeedCatCorpusParser.PairClass.FAKE_SATIRE: ("v", "fake-satire", "#009999"),
                        #WebisBuzzfeedCatCorpusParser.PairClass.SATIRE_REAL: ("D", "satire-real", "#aa6600")
                        WebisBuzzfeedCatCorpusParser.PairClass.FAKE_LEFT_FAKE_RIGHT: ("<", "fake,left-fake,right", "#990000"),
                        WebisBuzzfeedCatCorpusParser.PairClass.FAKE_LEFT_REAL_LEFT: (">", "fake,left-real,left", "#eeaa00"),
                        WebisBuzzfeedCatCorpusParser.PairClass.FAKE_RIGHT_REAL_RIGHT: ("x", "fake,right-real,right", "#000099"),
                        WebisBuzzfeedCatCorpusParser.PairClass.REAL_RIGHT_REAL_LEFT: ("v", "real,right-real,left", "#009999"),
                    }
    
                    parser = WebisBuzzfeedCatCorpusParser("corpora/buzzfeed", chunk_tokenizer,
                                                          ["articles_buzzfeed1", "articles_buzzfeed2"],
                                                          WebisBuzzfeedCatCorpusParser.class_by_orientation_and_veracity)
                elif experiment == "portal_authorship":
                    labels = {
                        WebisBuzzfeedAuthorshipCorpusParser.Class.SAME_PORTAL: ("o", "same portal", None),
                        WebisBuzzfeedAuthorshipCorpusParser.Class.DIFFERENT_PORTALS: ("x", "different portals", None)
                    }
                    parser = WebisBuzzfeedAuthorshipCorpusParser("corpora/buzzfeed", chunk_tokenizer,
                                                                 ["articles_buzzfeed1"])
                else:
                    raise ValueError("Invalid experiment")
                
                curve_plotter = UnmaskingCurvePlotter(labels, (-.2, 1.0), True)
                EventBroadcaster.subscribe("onUnmaskingRoundFinished", curve_plotter)
                s = UniqueRandomUndersampler()
                for i, pair in enumerate(parser):
                    fs = AvgWordFreqFeatureSet(pair, s)
                    #fs = AvgCharNgramFreqFeatureSet(pair, s, 3)
                    #fs = AvgDisjunctCharNgramFreqFeatureSet(pair, s, 3)
                    strat = FeatureRemoval(removed_per_round)
                    strat.run(iterations, num_features, fs, False)
                EventBroadcaster.unsubscribe("onUnmaskingRoundFinished", curve_plotter)
                
                stats_accumulator.set_meta_data({
                    "removed_per_round": removed_per_round,
                    "iterations": iterations,
                    "num_features": num_features,
                    "chunk_tokenizer": chunk_tokenizer.__class__.__name__
                })
                save_output(curve_plotter, stats_accumulator)
                stats_accumulator.reinit()
                curve_plotter.close()
            
            # noinspection PyUnboundLocalVariable
            curve_plotter = UnmaskingCurvePlotter(labels, (-.2, 1.0), True)
            curve_plotter.set_plot_title("Average curves after {} runs".format(num_experiments))
            avg_curves = curve_averager.get_avg_curves()
            for c in avg_curves:
                curve_id = curve_plotter.start_new_curve()
                curve_plotter.plot_curve(avg_curves[c], (0.0, len(avg_curves[c]) - 1), c, curve_id)
            curve_plotter.save("out/unmasking_" + str(int(time())) + "_avg.svg")
            
        elif corpus == "gutenberg_test":
            curve_plotter = UnmaskingCurvePlotter({
                BookSampleParser.Class.SAME_AUTHOR: ("o", "same author", None),
                BookSampleParser.Class.DIFFERENT_AUTHORS: ("x", "different authors", None)
            })
            EventBroadcaster.subscribe("onUnmaskingRoundFinished", curve_plotter)
            
            chunk_tokenizer = SentenceChunkTokenizer(500)
            parser = BookSampleParser("corpora/gutenberg_test", chunk_tokenizer)
            s = UniqueRandomUndersampler()
    
            chunking_progress = None
            for i, pair in enumerate(parser):
                if chunking_progress is not None:
                    EventBroadcaster.unsubscribe("onProgress", chunking_progress, {SamplePair})
    
                chunking_progress = ProgressPrinter("Chunking progress for pair {}".format(i))
                EventBroadcaster.subscribe("onProgress", chunking_progress, {SamplePair})
    
                fs = AvgWordFreqFeatureSet(pair, s)
                strat = FeatureRemoval(removed_per_round)
                strat.run(iterations, num_features, fs, False)

            stats_accumulator.set_meta_data({
                "removed_per_round": removed_per_round,
                "iterations": iterations,
                "num_features": num_features,
                "chunk_tokenizer": chunk_tokenizer.__class__.__name__
            })
            save_output(curve_plotter, stats_accumulator)
        else:
            raise ValueError("Invalid corpus")
        
        print("Time taken: {:.03f} seconds.".format(time() - start_time))
        
        # block, so window doesn't close automatically
        input("Press enter to terminate...")
    except KeyboardInterrupt:
        print("Exited upon user request.")
        exit(1)

if __name__ == "__main__":
    main()
