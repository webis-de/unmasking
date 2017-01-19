#!/usr/bin/env python3

from input import BookSampleParser
from classifier import UniqueRandomUndersampler, AvgWordFreqFeatureSet


def main():
    parser = BookSampleParser("corpora", 500, "english")
    s = UniqueRandomUndersampler()
    for pair in parser:
        fs = AvgWordFreqFeatureSet(pair, s)
        print((fs.get_features_absolute(20)))

if __name__ == "__main__":
    main()
